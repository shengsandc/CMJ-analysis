#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:20:37 2023

@author: sheng
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'


# import files from 'Result'
result_directory = '/Users/sheng/Library/Mobile Documents/com~apple~CloudDocs/Desktop/S&C/CMJ data/Result'
files = glob.glob(f"{result_directory}/*.csv")
dataframes = [pd.read_csv(file) for file in files]

# merge data
concat_data = pd.concat(dataframes, ignore_index = True).sort_values('date')

performance_var = concat_data.columns.drop(['name','date'])

# Variable description
var_desc={'jump_height':'垂直跳的跳躍高度',
          'modRSI':'動態肌力指標：跳躍高度除以跳躍時間，亦指彈速。彈速的進步可以來自於跳躍高度的成長或是跳躍時間的減少',
          'propulsive_RFD_100ms':'向心階段的發力率（100ms）：以0.1秒為區間去看向心階段反作用力提升幅度的最大值，它代表的是力量的斜率，可以想像在跳躍的時候力量的走向像是一個Ｖ字型，RFD好的選手的Ｖ會非常陡峭，不好的選手則會是很平滑，我們要盡可能提升力量的斜率，更趨近於陡峭的Ｖ',
          'mean_force_propulsive':'向心階段的平均力：平均力量越高越好，這個指標撇除了速度，僅觀察我們能夠累積給地面的力量有多大',
          'peak_force_propulsive':'向心階段的力量峰值：它代表的是整個跳躍過程我們能夠給地面最大的反作用力，跟最大肌力有極高的相關性',
          'pr_mean_power': '向心階段的平均功率：這個指標就納入速度了，功率的算法是力量X速度，可以說是爆發力的代表性指標'}

pdf_filename='CMJ Report.pdf'

# create a separate figure for each variable
with PdfPages(pdf_filename) as pdf:
    # create a separate figure for each variable
    for var in performance_var:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use different line styles, markers, and colors for each athlete
        styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D']
        colors = sns.color_palette('husl', n_colors=len(concat_data['name'].unique()))

        for i, athlete in enumerate(concat_data['name'].unique()):
            athlete_data = concat_data[concat_data['name'] == athlete]
            style = styles[i % len(styles)]
            marker = markers[i % len(markers)]
            color = colors[i]

            sns.lineplot(x='date', y=var, data=athlete_data, label=athlete, linestyle=style, marker=marker, color=color)

            # Annotate data labels at specific points (e.g., last data point)
            last_point = athlete_data.iloc[-1]
            plt.annotate(f'{last_point[var]:.2f}', (last_point['date'], last_point[var]),
                         textcoords="offset points", xytext=(-10, 5), ha='center', fontsize=8, color=color)

        ax.set_title(f'{var} Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel(var)

        # Move the legend outside the plot
        ax.legend(title='Athlete', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add description to the plot
        wrapped_description = textwrap.fill(var_desc.get(var, ""), width=60)
        ax.annotate(wrapped_description, xy=(0.5, -0.2), ha='center', va='center', fontsize=10,
                    xycoords='axes fraction', wrap=True)

        plt.subplots_adjust(bottom=0.65)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"Plots save to {pdf_filename}")
        
        
        
        
        
        
        
        
        