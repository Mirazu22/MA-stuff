# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:54:53 2025

@author: maxuh
"""

def transform_occupation_rates(df, rate_prefixes=None, occ_col='occ4'):
    """
    Transform wide occupation rate data to long format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Wide DataFrame with occupation codes as index
    rate_prefixes : list, optional
        List of rate column prefixes to process
    occ_col : str
        Name for occupation code column
        
    Returns:
    --------
    tuple of pandas.DataFrame
        Long format DataFrames for each rate type
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
            continue
            
        try:
            df_long = (
                df_wide
                .melt(id_vars=occ_col, value_vars=cols,
                      var_name='year', value_name=f'delta_{prefix.rstrip("_")}_hat')
                .assign(year=lambda d: pd.to_numeric(
                    d['year'].str.extract(r'(\d{4})')[0], errors='coerce'))
                .dropna(subset=['year'])
                .astype({'year': int})
                .sort_values([occ_col, 'year'])
                .reset_index(drop=True)
            )
            results[prefix.rstrip('_')] = df_long
        except Exception as e:
            print(f"Warning: Failed to process {prefix} columns: {e}")
    
    return results