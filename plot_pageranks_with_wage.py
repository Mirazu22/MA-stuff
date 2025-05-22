# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:38:49 2025

@author: maxuh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualization_acc_trans(
    accessibility,
    transferability,
    occupation_names=None,
    wages=None,
    figsize=(12, 10),
    cmap='viridis'
):
    """
    Create a log-log scatterplot of accessibility vs transferability,
    coloring each point by its (log-scale) wage.

    Parameters:
    -----------
    accessibility : array-like
        Accessibility scores.
    transferability : array-like
        Transferability scores.
    occupation_names : list of str, optional
        Names of occupations for labeling.
    wages : array-like, optional
        Wages (already in log-scale). Used to color the points.
    figsize : tuple, optional
        Figure size.
    cmap : str or Colormap, optional
        Matplotlib colormap name for wages.
    """
    # build DataFrame
    df = pd.DataFrame({
        'Accessibility': accessibility,
        'Transferability': transferability
    })
    if wages is not None:
        df['Wage'] = wages

    if occupation_names is not None:
        df['Occupation'] = occupation_names

    # start plot
    plt.figure(figsize=figsize)

    # scatter, colored by wage if provided
    if wages is not None:
        vmin, vmax = df['Wage'].min(), df['Wage'].max()
        sc = plt.scatter(
            df['Transferability'],
            df['Accessibility'],
            c=df['Wage'],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=100,
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        cbar = plt.colorbar(sc)
        cbar.set_label('Log Wage', rotation=270, labelpad=15)
    else:
        plt.scatter(
            df['Transferability'],
            df['Accessibility'],
            s=100,
            alpha=0.7,
            color='gray',
            edgecolors='w',
            linewidth=0.5
        )

    # log scales
    plt.xscale('log')
    plt.yscale('log')

    # optional labeling: top-5 by combined score
    if occupation_names is not None:
        df['Combined'] = df['Accessibility'] + df['Transferability']
        top5 = df.nlargest(5, 'Combined').index
        for idx in top5:
            x, y = df.loc[idx, ['Transferability', 'Accessibility']]
            plt.annotate(
                df.loc[idx, 'Occupation'],
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

    plt.title('Occupation Space: Accessibility vs Transferability', fontsize=14)
    plt.xlabel('Transferability (log scale)', fontsize=12)
    plt.ylabel('Accessibility (log scale)', fontsize=12)
    plt.tight_layout()
    plt.show()
