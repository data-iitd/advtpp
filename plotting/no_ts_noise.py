import numpy as np
import pickle
import pandas as pd
import itertools
import pdb
import datetime
from datetime import timezone
import time
import sys, os
import glob, subprocess
import random
import numpy as np
import pickle
import seaborn as sns
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 3, 3
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath,amsfonts} \usepackage{bm} \boldmath"

def plot_bar(means, stds, v1, name, text, dtype):
	ind = np.arange(len(means))*0.12
	colors = list(mcolors.TABLEAU_COLORS)
	colors.pop(1)
	colors.pop(1)
	colors.pop(1)
	colors.pop(1)
	colors.pop(1)
	# plt.text(0.2, 0.1, r'\textbf{100}', fontsize=20)
	plt.figure()
	# if name == 'elec_hinge':
	# 	plt.text(-0.02, v1, r'\textbf{'+text+'}', fontsize=16, bbox=dict(facecolor='white', alpha=1))
	# else:
	# 	plt.text(0, v1, r'\textbf{'+text+'}', fontsize=16, bbox=dict(facecolor='white', alpha=1))
	
	# plt.ylim(-10, 110)
	plt.locator_params(axis='y', nbins=5)
	# plt.bar(ind, means, color=colors, align='center', yerr=stds, ecolor='k', error_kw=dict(lw=2, capsize=10, capthick=0), linewidth=0, edgecolor='k', width = 0.1)
	plt.bar(ind, means, color=colors, align='center', linewidth=1, edgecolor='k', width = 0.1)
	# plt.xticks(ind,[r'\textbf{GPT-4-Vision}', r'\textbf{LLaVA}', r'\textbf{Mini-GPT}'], fontsize=16)
	plt.xticks(ind,[r'', r''], fontsize=20)
	# plt.ylabel(r'\textbf{Accuracy (\%)}$\rightarrow$', fontsize=16)

	if dtype == 'mae':
		plt.ylabel(r'\textbf{MAE} $\rightarrow$', fontsize=20)
	else:
		plt.ylabel(r'\textbf{MPA} $\rightarrow$', fontsize=20)


	plt.yticks(fontsize=16)
	plt.savefig(name+"_"+dtype+".pdf", bbox_inches='tight', pad_inches = 0)


def legend_plot():
	patterns = ["\\" , "/" , "+" , "-", ".", "*","x", "o", "O"]
	colors = list(mcolors.TABLEAU_COLORS)
	colors.pop(1)
	colors.pop(1)
	colors.pop(1)
	colors.pop(1)
	colors.pop(1)

	x = 2
	y = 3
	fig = plt.figure("Line plot")
	legendFig = plt.figure("Legend plot")
	ax = fig.add_subplot(111)
	line0, = ax.bar(x, y, facecolor=colors[0], edgecolor='k')
	line1, = ax.bar(x, y, facecolor=colors[1], edgecolor='k')
	ax.legend()
	legendFig.legend([line0, line1], 
		[r'\textbf{w/ Time-Noise}', r'\textbf{w/o Time-Noise}'], loc='center', prop={'size': 28}, frameon=False, ncol=6)
	legendFig.savefig('legend_nots.pdf', bbox_inches='tight', pad_inches = 0.01)

data_1 = [9.21, 13.27]
v1 = min(data_1)*0.1
name = "taobao_no_ts"
text = 'Taobao'
plot_bar(data_1, [], v1, name, text, 'mpa')
data_1 = [1.35065, 0.98136]
v1 = min(data_1)*0.1
plot_bar(data_1, [], v1, name, text, 'mae')


data_1 = [55.84, 58.35]
v1 = min(data_1)*0.1
name = "twitter_no_ts"
text = 'Twitter'

plot_bar(data_1, [], v1, name, text, 'mpa')
data_1 = [0.12084, 0.1006]
v1 = min(data_1)*0.1
plot_bar(data_1, [], v1, name, text, 'mae')


data_1 = [50.31, 54.04]
v1 = min(data_1)*0.1
name = "health_no_ts"
text = 'Health'
plot_bar(data_1, [], v1, name, text, 'mpa')
data_1 = [0.23991, 0.07922]
v1 = min(data_1)*0.1
plot_bar(data_1, [], v1, name, text, 'mae')


data_1 = [87.96, 94.60]
v1 = min(data_1)*0.1
name = "elec_no_ts"
text = 'Electricity'
plot_bar(data_1, [], v1, name, text, 'mpa')
data_1 = [0.07848, 0.02406]
v1 = min(data_1)*0.1
plot_bar(data_1, [], v1, name, text, 'mae')


legend_plot()
