# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:36:25 2025

@author: maxuh
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln
import warnings

# --- Step 2: Time smoothing per occupation ---
def smooth_over_time(df, rate_col, occupation_col, year_col, lower_q=0.25, upper_q=0.75, trim_fraction=0.1):
    """
    1) Compute median and IQR per occupation.
    2) Compute trimmed mean per occupation.
    3) Replace annual rate outside median ± 3*IQR by trimmed mean.
    4) Return a DataFrame with one row per occupation and a time-smoothed rate.
    """
    out = []
    for occ, grp in df.groupby(occupation_col):
        rates = grp[rate_col].dropna()
        if len(rates) == 0:
            continue
        
        median = rates.median()
        q1, q3 = rates.quantile([lower_q, upper_q])
        iqr = q3 - q1
        
        # Handle case where IQR is zero
        if iqr == 0:
            iqr = rates.std() if rates.std() > 0 else 0.01
        
        # trimmed mean
        sorted_rates = rates.sort_values()
        n = len(sorted_rates)
        k = int(np.floor(trim_fraction * n))
        trimmed = sorted_rates.iloc[k: n - k] if n > 2*k else sorted_rates
        trim_mean = trimmed.mean()
        
        # replace outliers
        def replace_outlier(r):
            if r < median - 3 * iqr or r > median + 3 * iqr:
                return trim_mean
            return r
        
        cleaned_rates = rates.apply(replace_outlier)
        final = cleaned_rates.mean()
        
        out.append({
            occupation_col: occ, 
            rate_col + '_time_smoothed': final
        })
    
    return pd.DataFrame(out)

# --- Step 3: Empirical-Bayes shrinkage per 1-digit group ---
def fit_beta_prior(S, N):
    """Fit Beta(a, b) prior by maximizing marginal likelihood, assuming a=b."""
    # Validate inputs
    if len(S) == 0 or np.any(N <= 0) or np.any(S < 0) or np.any(S > N):
        warnings.warn("Invalid data for beta fitting, using default priors")
        return 1.0, 1.0
    
    def neg_marginal_loglik(log_ab):
        a = np.exp(log_ab)
        b = a
        try:
            # per-obs contribution: betaln(S+a, N-S+b) - betaln(a,b)
            loglik = np.sum(betaln(S + a, N - S + b) - betaln(a, b))
            return -loglik
        except (ValueError, OverflowError):
            return np.inf
    
    # optimize in log-space with bounds
    try:
        res = minimize_scalar(neg_marginal_loglik, bounds=(-5, 5), method='bounded')
        if res.success:
            a_hat = np.exp(res.x)
            b_hat = a_hat
        else:
            a_hat, b_hat = 1.0, 1.0
    except:
        a_hat, b_hat = 1.0, 1.0
    
    return a_hat, b_hat

def eb_shrinkage(df, count_col, total_col, group_col, occupation_col):
    """
    Compute EB posterior means per occupation, fitting separate Beta priors for each group.
    df must have per-occupation aggregated counts and totals.
    Returns DataFrame with occupation_col and eb_shrunk_rate.
    """
    results = []
    
    for grp_name, grp in df.groupby(group_col):
        if len(grp) == 0:
            continue
            
        S = grp[count_col].values
        N = grp[total_col].values
        
        # Validate data
        valid_mask = (N > 0) & (S >= 0) & (S <= N)
        if not np.any(valid_mask):
            continue
            
        S_valid = S[valid_mask]
        N_valid = N[valid_mask]
        
        # fit group-specific prior
        a, b = fit_beta_prior(S_valid, N_valid)
        
        # compute posterior mean per occupation
        for _, row in grp.iterrows():
            S_i = row[count_col]
            N_i = row[total_col]
            
            if N_i > 0 and S_i >= 0 and S_i <= N_i:
                post_mean = (S_i + a) / (N_i + a + b)
            else:
                post_mean = np.nan
                
            results.append({
                occupation_col: row[occupation_col],
                group_col: grp_name,
                'eb_shrunk_rate': post_mean
            })
    
    return pd.DataFrame(results)

# --- Example usage ---
def process_data(df_u, df_v):
    """
    Complete processing pipeline for unemployment and vacancy data.
    
    Expected columns:
    df_u: ['occ4', 'occ1', 'year', 'S', 'E', 'delta_u_hat']
    df_v: ['occ4', 'occ1', 'year', 'J', 'E', 'delta_v_hat']
    """
    
    # Step 2: time smoothing
    df_u_smoothed = smooth_over_time(df_u, rate_col='delta_u_hat', 
                                   occupation_col='occ4', year_col='year')
    df_v_smoothed = smooth_over_time(df_v, rate_col='delta_v_hat', 
                                   occupation_col='occ4', year_col='year')
    
    # Step 3: aggregate counts across years for EB shrinkage
    agg_u = df_u.groupby(['occ4', 'occ1']).agg({
        'S': 'sum',
        'E': 'sum'
    }).reset_index()
    
    agg_v = df_v.groupby(['occ4', 'occ1']).agg({
        'J': 'sum', 
        'E': 'sum'
    }).reset_index()
    
    # Apply EB shrinkage
    eb_u = eb_shrinkage(agg_u, count_col='S', total_col='E', 
                       group_col='occ1', occupation_col='occ4')
    eb_v = eb_shrinkage(agg_v, count_col='J', total_col='E', 
                       group_col='occ1', occupation_col='occ4')
    
    # Merge time-smoothed and EB-shrunk rates
    final_u = df_u_smoothed.merge(eb_u, on='occ4', how='left')
    final_v = df_v_smoothed.merge(eb_v, on='occ4', how='left')
    
    return final_u, final_v

# Test function
def test_with_sample_data():
    """Generate sample data to test the functions."""
    np.random.seed(42)
    
    # Create sample unemployment data
    occupations = [f'occ_{i}' for i in range(1, 21)]
    occ1_mapping = {occ: f'group_{i//5}' for i, occ in enumerate(occupations)}
    
    data_u = []
    for occ in occupations:
        for year in range(2018, 2023):
            E = np.random.randint(100, 1000)
            S = np.random.binomial(E, 0.05)  # 5% unemployment rate
            delta_u_hat = S / E + np.random.normal(0, 0.01)
            data_u.append({
                'occ4': occ,
                'occ1': occ1_mapping[occ],
                'year': year,
                'S': S,
                'E': E,
                'delta_u_hat': delta_u_hat
            })
    
    # Create sample vacancy data
    data_v = []
    for occ in occupations:
        for year in range(2018, 2023):
            E = np.random.randint(100, 1000)
            J = np.random.poisson(E * 0.03)  # 3% vacancy rate
            delta_v_hat = J / E + np.random.normal(0, 0.005)
            data_v.append({
                'occ4': occ,
                'occ1': occ1_mapping[occ],
                'year': year,
                'J': J,
                'E': E,
                'delta_v_hat': delta_v_hat
            })
    
    df_u = pd.DataFrame(data_u)
    df_v = pd.DataFrame(data_v)
    
    # Process the data
    final_u, final_v = process_data(df_u, df_v)
    
    print("Sample results:")
    print("\nUnemployment data (first 5 rows):")
    print(final_u.head())
    print("\nVacancy data (first 5 rows):")
    print(final_v.head())
    
    return final_u, final_v

# Uncomment to test:
test_with_sample_data()