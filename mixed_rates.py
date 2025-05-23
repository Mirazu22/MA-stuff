# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:28:23 2025

@author: maxuh
"""

import pandas as pd
import numpy as np

def calculate_mixed_occupation_rates(df_wide, df_long, rate_type, top_n=200):
    """
    Calculate mixed occupation rates using top N occupations' own rates and 
    job family aggregates for others.
    
    Parameters:
    -----------
    df_wide : pandas.DataFrame
        Wide DataFrame with occ_code as index and max_emp column
    df_long : pandas.DataFrame  
        Long DataFrame with columns ['occ_code', 'year', rate_column]
    rate_type : str
        Rate type ('alo_rate' or 'vac_rate')
    top_n : int
        Number of top occupations to use individual rates for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with occ_code as index and calculated rates
    """
    
    # Validate inputs
    if 'max_emp' not in df_wide.columns:
        raise ValueError("df_wide must contain 'max_emp' column")
    
    rate_column = f'delta_{rate_type}_hat'
    if rate_column not in df_long.columns:
        raise ValueError(f"df_long must contain '{rate_column}' column")
    
    # Get top N occupations by employment
    top_occupations = df_wide.nlargest(top_n, 'max_emp').index.tolist()
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=df_wide.index)
    results.index.name = 'occ_code'
    
    # Calculate mean and last (2023) rates for all occupations
    occ_stats = (
        df_long.groupby('occ_code')[rate_column]
        .agg(['mean', 'count'])
        .rename(columns={'mean': f'{rate_type}_mean', 'count': f'{rate_type}_count'})
    )
    
    # Get 2023 rates
    df_2023 = df_long[df_long['year'] == 2023].set_index('occ_code')[rate_column]
    if df_2023.empty:
        raise ValueError("No data found for year 2023")
    
    # Check for missing 2023 data
    missing_2023 = set(df_wide.index) - set(df_2023.index)
    if missing_2023:
        raise ValueError(f"Missing 2023 data for occupations: {sorted(list(missing_2023))}")
    
    occ_stats[f'{rate_type}_2023'] = df_2023
    
    # Check for occupations with no data
    missing_data = set(df_wide.index) - set(occ_stats.index)
    if missing_data:
        raise ValueError(f"No rate data found for occupations: {sorted(list(missing_data))}")
    
    # Merge with results
    results = results.join(occ_stats, how='left')
    
    # Create job family columns
    results['occ_code_str'] = results.index.astype(str).str.zfill(4)
    results['job_family_2d'] = results['occ_code_str'].str[:2]
    results['job_family_1d'] = results['occ_code_str'].str[:1]
    
    # Merge employment weights
    results = results.join(df_wide[['max_emp']], how='left')
    
    # Calculate 2-digit job family aggregates (weighted by max_emp)
    family_2d_stats = []
    for family in results['job_family_2d'].unique():
        family_occs = results[results['job_family_2d'] == family]
        if len(family_occs) == 0:
            continue
            
        # Weighted mean calculation
        weights = family_occs['max_emp']
        mean_weighted = np.average(family_occs[f'{rate_type}_mean'], weights=weights)
        rate_2023_weighted = np.average(family_occs[f'{rate_type}_2023'], weights=weights)
        
        family_2d_stats.append({
            'job_family_2d': family,
            f'{rate_type}_mean_2d': mean_weighted,
            f'{rate_type}_2023_2d': rate_2023_weighted
        })
    
    family_2d_df = pd.DataFrame(family_2d_stats).set_index('job_family_2d')
    results = results.join(family_2d_df, on='job_family_2d', how='left')
    
    # Calculate 1-digit job family aggregates (weighted by max_emp)
    family_1d_stats = []
    for family in results['job_family_1d'].unique():
        family_occs = results[results['job_family_1d'] == family]
        if len(family_occs) == 0:
            continue
            
        # Weighted mean calculation
        weights = family_occs['max_emp']
        mean_weighted = np.average(family_occs[f'{rate_type}_mean'], weights=weights)
        rate_2023_weighted = np.average(family_occs[f'{rate_type}_2023'], weights=weights)
        
        family_1d_stats.append({
            'job_family_1d': family,
            f'{rate_type}_mean_1d': mean_weighted,
            f'{rate_type}_2023_1d': rate_2023_weighted
        })
    
    family_1d_df = pd.DataFrame(family_1d_stats).set_index('job_family_1d')
    results = results.join(family_1d_df, on='job_family_1d', how='left')
    
    # Check for missing aggregates
    missing_2d = results[results[f'{rate_type}_mean_2d'].isna()]['job_family_2d'].unique()
    if len(missing_2d) > 0:
        raise ValueError(f"Missing 2-digit aggregates for families: {list(missing_2d)}")
    
    missing_1d = results[results[f'{rate_type}_mean_1d'].isna()]['job_family_1d'].unique()
    if len(missing_1d) > 0:
        raise ValueError(f"Missing 1-digit aggregates for families: {list(missing_1d)}")
    
    # Create final mixed columns
    # For mean rates
    results[f'{rate_type}_final_mean'] = np.where(
        results.index.isin(top_occupations),
        results[f'{rate_type}_mean'],  # Use own mean for top occupations
        results[f'{rate_type}_mean_2d']  # Use 2-digit aggregate for others (preferential)
    )
    
    # For 2023 rates  
    results[f'{rate_type}_final_2023'] = np.where(
        results.index.isin(top_occupations),
        results[f'{rate_type}_2023'],  # Use own 2023 for top occupations
        results[f'{rate_type}_2023_2d']  # Use 2-digit aggregate for others (preferential)
    )
    
    # Clean up temporary columns
    cols_to_drop = ['occ_code_str', 'job_family_2d', 'job_family_1d', 'max_emp']
    results = results.drop(columns=cols_to_drop)
    
    return results

# Usage examples:
# For unemployment rates
df_alo_results = calculate_mixed_occupation_rates(df, df_u_long, 'alo_rate')

# For vacancy rates  
df_vac_results = calculate_mixed_occupation_rates(df, df_v_long, 'vac_rate')