# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:36:25 2025

@author: maxuh
"""

"""
Time smoothing and Empirical Bayes shrinkage for occupational data analysis.

This module provides functions for:
1. Time-based smoothing of rates per occupation using outlier detection and trimmed means
2. Empirical Bayes shrinkage using Beta priors fitted via marginal likelihood maximization
"""

import logging
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, OptimizeResult
from scipy.special import betaln

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def _validate_dataframe_columns(df: pd.DataFrame, required_columns: list, 
                               function_name: str) -> None:
    """Validate that DataFrame contains required columns."""
    if df.empty:
        raise DataValidationError(f"{function_name}: DataFrame is empty")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"{function_name}: Missing required columns: {missing_cols}"
        )


def _validate_numeric_column(df: pd.DataFrame, column: str, 
                           function_name: str, allow_negative: bool = True) -> None:
    """Validate that a column contains numeric data."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise DataValidationError(
            f"{function_name}: Column '{column}' must be numeric"
        )
    
    if not allow_negative and (df[column] < 0).any():
        raise DataValidationError(
            f"{function_name}: Column '{column}' cannot contain negative values"
        )


def smooth_over_time(df: pd.DataFrame, 
                    rate_col: str, 
                    occupation_col: str, 
                    year_col: str,
                    lower_q: float = 0.25, 
                    upper_q: float = 0.75, 
                    trim_fraction: float = 0.1,
                    outlier_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Apply time-based smoothing to rates per occupation using outlier detection.
    
    Algorithm:
    1. Compute median and IQR per occupation across years
    2. Compute trimmed mean per occupation
    3. Replace annual rates outside median ± outlier_multiplier*IQR with trimmed mean
    4. Return time-smoothed rate as mean of cleaned rates
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with time series per occupation
    rate_col : str
        Column name containing rates to smooth
    occupation_col : str
        Column name for occupation identifier
    year_col : str
        Column name for time dimension
    lower_q : float, default 0.25
        Lower quantile for IQR calculation
    upper_q : float, default 0.75
        Upper quantile for IQR calculation
    trim_fraction : float, default 0.1
        Fraction to trim from each end for trimmed mean
    outlier_multiplier : float, default 3.0
        Multiplier for IQR-based outlier detection
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with occupation and time-smoothed rate columns
        
    Raises:
    -------
    DataValidationError
        If input validation fails
    ValueError
        If parameters are invalid
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    required_cols = [rate_col, occupation_col, year_col]
    _validate_dataframe_columns(df, required_cols, "smooth_over_time")
    _validate_numeric_column(df, rate_col, "smooth_over_time")
    
    # Parameter validation
    if not (0 <= lower_q < upper_q <= 1):
        raise ValueError("lower_q must be < upper_q and both in [0,1]")
    if not (0 <= trim_fraction < 0.5):
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if outlier_multiplier <= 0:
        raise ValueError("outlier_multiplier must be positive")
    
    results = []
    processed_occupations = 0
    
    for occupation, group in df.groupby(occupation_col):
        rates = group[rate_col].dropna()
        
        if len(rates) == 0:
            logger.warning(f"No valid rates for occupation {occupation}, skipping")
            continue
            
        if len(rates) == 1:
            # Single observation - no smoothing needed
            smoothed_rate = rates.iloc[0]
        else:
            # Calculate statistics
            median_rate = rates.median()
            q1, q3 = rates.quantile([lower_q, upper_q])
            iqr = q3 - q1
            
            # Calculate trimmed mean
            sorted_rates = rates.sort_values()
            n = len(sorted_rates)
            k = max(0, int(np.floor(trim_fraction * n)))
            
            if n > 2 * k:
                trimmed_rates = sorted_rates.iloc[k:n-k]
            else:
                trimmed_rates = sorted_rates
                
            trim_mean = trimmed_rates.mean()
            
            # Replace outliers with trimmed mean
            lower_bound = median_rate - outlier_multiplier * iqr
            upper_bound = median_rate + outlier_multiplier * iqr
            
            cleaned_rates = rates.where(
                (rates >= lower_bound) & (rates <= upper_bound),
                trim_mean
            )
            
            smoothed_rate = cleaned_rates.mean()
        
        results.append({
            occupation_col: occupation,
            f"{rate_col}_time_smoothed": smoothed_rate
        })
        processed_occupations += 1
    
    logger.info(f"Processed {processed_occupations} occupations for time smoothing")
    
    if not results:
        logger.warning("No valid occupations found for smoothing")
        return pd.DataFrame(columns=[occupation_col, f"{rate_col}_time_smoothed"])
    
    return pd.DataFrame(results)


def _fit_beta_prior(success_counts: np.ndarray, 
                   total_counts: np.ndarray) -> Tuple[float, float]:
    """
    Fit symmetric Beta(a, a) prior by maximizing marginal likelihood.
    
    Parameters:
    -----------
    success_counts : np.ndarray
        Array of success counts
    total_counts : np.ndarray
        Array of total counts
        
    Returns:
    --------
    Tuple[float, float]
        Fitted parameters (a, b) where a = b
        
    Raises:
    -------
    ValueError
        If optimization fails or inputs are invalid
    """
    if len(success_counts) != len(total_counts):
        raise ValueError("success_counts and total_counts must have same length")
    
    if np.any(success_counts < 0) or np.any(total_counts <= 0):
        raise ValueError("success_counts must be non-negative, total_counts must be positive")
    
    if np.any(success_counts > total_counts):
        raise ValueError("success_counts cannot exceed total_counts")
    
    def neg_marginal_loglik(log_a: float) -> float:
        """Negative log marginal likelihood for symmetric Beta prior."""
        a = np.exp(log_a)
        try:
            # Marginal likelihood contribution per observation
            log_ml = betaln(success_counts + a, total_counts - success_counts + a) - betaln(a, a)
            return -np.sum(log_ml)
        except (OverflowError, ValueError):
            return np.inf
    
    # Optimize in log-space for numerical stability
    try:
        result: OptimizeResult = minimize_scalar(
            neg_marginal_loglik, 
            bounds=(-10, 10), 
            method='bounded'
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
            
        a_hat = np.exp(result.x)
        
        # Sanity check
        if not (0.001 <= a_hat <= 1000):
            logger.warning(f"Fitted parameter a={a_hat:.3f} may be extreme")
            
        return a_hat, a_hat
        
    except Exception as e:
        raise ValueError(f"Failed to fit Beta prior: {str(e)}")


def empirical_bayes_shrinkage(df: pd.DataFrame,
                            count_col: str,
                            total_col: str,
                            group_col: str,
                            occupation_col: str) -> pd.DataFrame:
    """
    Apply Empirical Bayes shrinkage using group-specific Beta priors.
    
    For each group, fits a symmetric Beta(a, a) prior by maximizing marginal likelihood,
    then computes posterior means for each occupation within the group.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aggregated data with success/total counts per occupation
    count_col : str
        Column name for success counts
    total_col : str
        Column name for total counts
    group_col : str
        Column name for grouping variable (e.g., 1-digit occupation code)
    occupation_col : str
        Column name for occupation identifier
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with occupation, group, and empirical Bayes shrunken rate
        
    Raises:
    -------
    DataValidationError
        If input validation fails
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    required_cols = [count_col, total_col, group_col, occupation_col]
    _validate_dataframe_columns(df, required_cols, "empirical_bayes_shrinkage")
    _validate_numeric_column(df, count_col, "empirical_bayes_shrinkage", allow_negative=False)
    _validate_numeric_column(df, total_col, "empirical_bayes_shrinkage", allow_negative=False)
    
    # Check that counts don't exceed totals
    if (df[count_col] > df[total_col]).any():
        raise DataValidationError("count_col values cannot exceed total_col values")
    
    results = []
    processed_groups = 0
    
    for group_name, group_data in df.groupby(group_col):
        if len(group_data) == 0:
            continue
            
        success_counts = group_data[count_col].values
        total_counts = group_data[total_col].values
        
        try:
            # Fit group-specific Beta prior
            a, b = _fit_beta_prior(success_counts, total_counts)
            logger.info(f"Group {group_name}: fitted Beta({a:.3f}, {b:.3f}) prior")
            
            # Compute posterior means
            for _, row in group_data.iterrows():
                s_i = row[count_col]
                n_i = row[total_col]
                
                # Beta-Binomial posterior mean
                posterior_mean = (s_i + a) / (n_i + a + b)
                
                results.append({
                    occupation_col: row[occupation_col],
                    group_col: group_name,
                    f"{count_col}_eb_rate": posterior_mean
                })
            
            processed_groups += 1
            
        except ValueError as e:
            logger.error(f"Failed to process group {group_name}: {str(e)}")
            continue
    
    logger.info(f"Successfully processed {processed_groups} groups for EB shrinkage")
    
    if not results:
        logger.warning("No groups successfully processed")
        return pd.DataFrame(columns=[occupation_col, group_col, f"{count_col}_eb_rate"])
    
    return pd.DataFrame(results)


def process_occupational_data(df_rates: pd.DataFrame,
                            rate_col: str,
                            count_col: str,
                            total_col: str,
                            occupation_col: str,
                            group_col: str,
                            year_col: str,
                            **smoothing_kwargs) -> pd.DataFrame:
    """
    Complete pipeline: time smoothing + empirical Bayes shrinkage.
    
    Parameters:
    -----------
    df_rates : pd.DataFrame
        Raw time series data with rates per occupation-year
    rate_col : str
        Column with rates to smooth
    count_col : str
        Column with success counts for EB shrinkage
    total_col : str
        Column with total counts for EB shrinkage
    occupation_col : str
        Occupation identifier column
    group_col : str
        Grouping variable for EB priors
    year_col : str
        Time dimension column
    **smoothing_kwargs
        Additional arguments for smooth_over_time()
        
    Returns:
    --------
    pd.DataFrame
        Combined results with time-smoothed and EB-shrunken rates
    """
    logger.info("Starting occupational data processing pipeline")
    
    # Step 1: Time smoothing
    df_smoothed = smooth_over_time(
        df_rates, rate_col, occupation_col, year_col, **smoothing_kwargs
    )
    
    # Step 2: Aggregate counts for EB shrinkage
    df_aggregated = df_rates.groupby([occupation_col, group_col]).agg({
        count_col: 'sum',
        total_col: 'sum'
    }).reset_index()
    
    # Step 3: EB shrinkage
    df_eb = empirical_bayes_shrinkage(
        df_aggregated, count_col, total_col, group_col, occupation_col
    )
    
    # Step 4: Merge results
    final_df = df_smoothed.merge(df_eb, on=occupation_col, how='outer')
    
    logger.info(f"Pipeline complete. Final dataset has {len(final_df)} occupations")
    
    return final_df


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Sample data structure
    sample_data = []
    occupations = [f"occ_{i:02d}" for i in range(1, 21)]  # 20 occupations
    groups = ['A', 'B', 'C', 'D']  # 4 groups
    years = list(range(2018, 2023))  # 5 years
    
    for occ in occupations:
        group = np.random.choice(groups)
        base_rate = np.random.beta(2, 8)  # Base rate around 0.2
        
        for year in years:
            # Add some year-to-year variation
            annual_rate = max(0, base_rate + np.random.normal(0, 0.05))
            total_count = np.random.poisson(100) + 50
            success_count = np.random.binomial(total_count, annual_rate)
            
            sample_data.append({
                'occ4': occ,
                'occ1': group,
                'year': year,
                'rate': annual_rate,
                'S': success_count,
                'N': total_count
            })
    
    df_sample = pd.DataFrame(sample_data)
    
    try:
        # Test the complete pipeline
        result = process_occupational_data(
            df_sample,
            rate_col='rate',
            count_col='S',
            total_col='N',
            occupation_col='occ4',
            group_col='occ1',
            year_col='year'
        )
        
        print("Sample results:")
        print(result.head(10))
        print(f"\nProcessed {len(result)} occupations successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise