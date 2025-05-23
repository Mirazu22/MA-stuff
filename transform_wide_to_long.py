# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:54:53 2025

@author: maxuh
"""

import pandas as pd
import numpy as np
import warnings

def transform_occupation_rates(df, rate_prefixes=None, occ_col='occ4', validate_rates=True):
    """
    Transform wide occupation rate data to long format with data cleaning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Wide DataFrame with occupation codes as index
    rate_prefixes : list, optional
        List of rate column prefixes to process (default: ['alo_rate_', 'vac_rate_'])
    occ_col : str
        Name for occupation code column (default: 'occ4')
    validate_rates : bool
        Whether to perform rate validation and cleaning (default: True)
        
    Returns:
    --------
    dict
        Dictionary with rate type as keys and cleaned long format DataFrames as values
        
    Raises:
    -------
    ValueError
        If input DataFrame is empty or if rates contain invalid data types after cleaning
    """
    if rate_prefixes is None:
        rate_prefixes = ['alo_rate_', 'vac_rate_']
    
    # Validation
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    df_wide = df.reset_index().rename(columns={'index': occ_col})
    results = {}
    
    for prefix in rate_prefixes:
        cols = [c for c in df_wide.columns if c.startswith(prefix)]
        if not cols:
            warnings.warn(f"No columns found with prefix '{prefix}'")
            continue
            
        try:
            # Clean rates before melting for better performance
            if validate_rates:
                df_cleaned = _clean_rate_columns(df_wide.copy(), cols, prefix)
            else:
                df_cleaned = df_wide
            
            # Transform to long format
            rate_name = f'delta_{prefix.rstrip("_")}_hat'
            df_long = (
                df_cleaned
                .melt(id_vars=occ_col, value_vars=cols,
                      var_name='year_col', value_name=rate_name)
                .assign(year=lambda d: pd.to_numeric(
                    d['year_col'].str.extract(r'(\d{4})')[0], errors='coerce'))
                .drop(columns=['year_col'])
                .dropna(subset=['year'])
                .astype({'year': int})
                .sort_values([occ_col, 'year'])
                .reset_index(drop=True)
            )
            
            # Final validation of rate column
            if validate_rates:
                _validate_final_rates(df_long, rate_name)
            
            results[prefix.rstrip('_')] = df_long
            
        except Exception as e:
            raise RuntimeError(f"Failed to process {prefix} columns: {e}")
    
    return results

def _clean_rate_columns(df, rate_cols, prefix):
    """
    Clean rate columns by replacing invalid values with NaN.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing rate columns
    rate_cols : list
        List of rate column names to clean
    prefix : str
        Rate prefix for logging purposes
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cleaned rate columns
    """
    df_cleaned = df.copy()
    total_cleaned = 0
    
    for col in rate_cols:
        if col not in df_cleaned.columns:
            continue
            
        original_count = df_cleaned[col].notna().sum()
        
        # Replace infinite values with NaN
        inf_mask = np.isinf(df_cleaned[col])
        inf_count = inf_mask.sum()
        df_cleaned.loc[inf_mask, col] = np.nan
        
        # Replace values outside [0, 1] range with NaN
        # Only check numeric values that aren't already NaN
        numeric_mask = pd.to_numeric(df_cleaned[col], errors='coerce').notna()
        range_mask = numeric_mask & ((df_cleaned[col] < 0) | (df_cleaned[col] > 1))
        range_count = range_mask.sum()
        df_cleaned.loc[range_mask, col] = np.nan
        
        cleaned_count = inf_count + range_count
        total_cleaned += cleaned_count
        
        if cleaned_count > 0:
            remaining_valid = df_cleaned[col].notna().sum()
            warnings.warn(
                f"Column '{col}': Cleaned {cleaned_count} invalid values "
                f"({inf_count} infinite, {range_count} out of range [0,1]). "
                f"Valid values: {original_count} → {remaining_valid}"
            )
    
    if total_cleaned > 0:
        print(f"Total values cleaned for {prefix} columns: {total_cleaned}")
    
    return df_cleaned

def _validate_final_rates(df, rate_col):
    """
    Validate that rate column contains only numeric values or NaN.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate
    rate_col : str
        Name of rate column to validate
        
    Raises:
    -------
    ValueError
        If rate column contains non-numeric, non-NaN values
    """
    if rate_col not in df.columns:
        return
    
    # Check if all values are either numeric or NaN
    numeric_or_nan = pd.to_numeric(df[rate_col], errors='coerce')
    
    # Find values that couldn't be converted and aren't already NaN
    invalid_mask = (numeric_or_nan.isna()) & (df[rate_col].notna())
    
    if invalid_mask.any():
        invalid_values = df.loc[invalid_mask, rate_col].unique()
        raise ValueError(
            f"Rate column '{rate_col}' contains invalid non-numeric values after cleaning: "
            f"{invalid_values[:5]}{'...' if len(invalid_values) > 5 else ''}"
        )
    
    # Additional validation: check if cleaned rates are still in valid range
    valid_rates = numeric_or_nan.dropna()
    if len(valid_rates) > 0:
        out_of_range = ((valid_rates < 0) | (valid_rates > 1)).sum()
        if out_of_range > 0:
            raise ValueError(
                f"Rate column '{rate_col}' still contains {out_of_range} values outside [0,1] range after cleaning"
            )

# Example usage with the original transformation logic:
def clean_occupation_rates_original_style(df):
    """
    Original style transformation with added cleaning functionality.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Wide DataFrame indexed by 4-digit occupation code
        
    Returns:
    --------
    tuple
        (df_u_long, df_v_long) - cleaned long format DataFrames
    """
    # Step 0: bring the index into a column
    df_wide = df.reset_index().rename(columns={'index': 'occ4'})
    
    # Step 1: melt out the delta_u_hat columns with cleaning
    u_cols = [c for c in df_wide.columns if c.startswith('alo_rate_')]
    
    # Clean unemployment rate columns
    df_wide_cleaned = _clean_rate_columns(df_wide.copy(), u_cols, 'alo_rate_')
    
    df_u_long = (
        df_wide_cleaned
        .melt(id_vars='occ4', value_vars=u_cols,
              var_name='year', value_name='delta_u_hat')
        .assign(year=lambda d: pd.to_numeric(
            d['year'].str.extract(r'alo_rate_(\d{4})')[0], errors='coerce'))
        .dropna(subset=['year'])
        .astype({'year': int})
        .sort_values(['occ4','year'])
        .reset_index(drop=True)
    )
    
    # Validate cleaned unemployment rates
    _validate_final_rates(df_u_long, 'delta_u_hat')
    
    # Step 2: melt out the delta_v_hat columns with cleaning
    v_cols = [c for c in df_wide.columns if c.startswith('vac_rate_')]
    
    # Clean vacancy rate columns
    df_wide_cleaned = _clean_rate_columns(df_wide.copy(), v_cols, 'vac_rate_')
    
    df_v_long = (
        df_wide_cleaned
        .melt(id_vars='occ4', value_vars=v_cols,
              var_name='year', value_name='delta_v_hat')
        .assign(year=lambda d: pd.to_numeric(
            d['year'].str.extract(r'vac_rate_(\d{4})')[0], errors='coerce'))
        .dropna(subset=['year'])
        .astype({'year': int})
        .sort_values(['occ4','year'])
        .reset_index(drop=True)
    )
    
    # Validate cleaned vacancy rates
    _validate_final_rates(df_v_long, 'delta_v_hat')
    
    return df_u_long, df_v_long

# Usage examples:
if __name__ == "__main__":
    # Example 1: Using the enhanced function
    # results = transform_occupation_rates(df)
    # df_u_long = results['alo_rate']
    # df_v_long = results['vac_rate']
    
    # Example 2: Using original style with cleaning
    # df_u_long, df_v_long = clean_occupation_rates_original_style(df)
    pass