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

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 2, 3
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath,amsfonts} \usepackage{bm} \boldmath"

def plot_mae(means, stds, v1, name, text):
	ind = np.arange(len(means))*0.12
	colors = ["#0072B2", "sandybrown"]
	# plt.text(0.2, 0.1, r'\textbf{100}', fontsize=20)
	plt.figure()
	if name == 'elec_hinge':
		plt.text(-0.02, v1, r'\textbf{'+text+'}', fontsize=16, bbox=dict(facecolor='white', alpha=1))
	else:
		plt.text(0, v1, r'\textbf{'+text+'}', fontsize=16, bbox=dict(facecolor='white', alpha=1))
	# if name == 'RQ1':
	# 	plt.text(-0.2, 101, r'\textbf{100}', fontsize=14)
	# 	plt.text(0.80, 43.8, r'\textbf{42.8}', fontsize=14)
	# 	plt.text(1.9, 1, r'\textbf{0}', fontsize=14)

	# elif name == 'RQ2':
	# 	plt.text(-0.2, 101, r'\textbf{100}', fontsize=14)
	# 	plt.text(0.80, 101, r'\textbf{100}', fontsize=14)
	# 	plt.text(1.9, 1, r'\textbf{N/A}', fontsize=14)

	# elif name == 'RQ3':
	# 	plt.text(-0.2, 43.8, r'\textbf{42.8}', fontsize=14)
	# 	plt.text(0.80, 1, r'\textbf{0}', fontsize=14)
	# 	plt.text(1.9, 1, r'\textbf{0}', fontsize=14)

	# elif name == 'RQ5':
	# 	plt.text(-0.2, 101, r'\textbf{100}', fontsize=14)
	# 	plt.text(0.80, 101, r'\textbf{100}', fontsize=14)
	# 	plt.text(1.9, 51, r'\textbf{50}', fontsize=14)

	# elif name == 'RQ6':
	# 	plt.text(-0.2, 101, r'\textbf{100}', fontsize=14)
	# 	plt.text(0.80, 101, r'\textbf{100}', fontsize=14)
	# 	plt.text(1.9, 101, r'\textbf{100}', fontsize=14)
	
	# plt.ylim(-10, 110)
	plt.locator_params(axis='y', nbins=5)
	# plt.bar(ind, means, color=colors, align='center', yerr=stds, ecolor='k', error_kw=dict(lw=2, capsize=10, capthick=0), linewidth=0, edgecolor='k', width = 0.1)
	plt.bar(ind, means, color=colors, align='center', linewidth=1, edgecolor='k', width = 0.1)
	# plt.xticks(ind,[r'\textbf{GPT-4-Vision}', r'\textbf{LLaVA}', r'\textbf{Mini-GPT}'], fontsize=16)
	plt.xticks(ind,[r'', r''], fontsize=20)
	# plt.ylabel(r'\textbf{Accuracy (\%)}$\rightarrow$', fontsize=16)
	plt.ylabel(r'\textbf{MAE} $\rightarrow$', fontsize=20)
	plt.yticks(fontsize=16)
	plt.savefig(name+".pdf", bbox_inches='tight', pad_inches = 0)


def legend_plot():
	patterns = ["\\" , "/" , "+" , "-", ".", "*","x", "o", "O"]
	colors = ["#0072B2", "sandybrown"]

	x = 2
	y = 3
	fig = plt.figure("Line plot")
	legendFig = plt.figure("Legend plot")
	ax = fig.add_subplot(111)
	line0, = ax.bar(x, y, facecolor=colors[0], edgecolor='k')
	line1, = ax.bar(x, y, facecolor=colors[1], edgecolor='k')
	ax.legend()
	legendFig.legend([line0, line1], 
		[r'\textbf{w/ Hinge}', r'\textbf{w/o Hinge}'], loc='center', prop={'size': 28}, frameon=False, ncol=6)
	legendFig.savefig('legend_hinge.pdf', bbox_inches='tight', pad_inches = 0.01)

data_1 = [1.332495, 1.374935]
err = [0.018155, 0.005185]
v1 = min(data_1)*0.1
name = "taobao_hinge"
text = 'Taobao'
plot_mae(data_1, err, v1, name, text)

data_1 = [0.21484, 0.18763]
err = [0.01573, 0.02738]
v1 = min(data_1)*0.1
name = "health_hinge"
text = 'Health'
plot_mae(data_1, err, v1, name, text)

data_1 = [0.07761, 0.083045]
err = [0.00087, 0.000325]
v1 = min(data_1)*0.1
name = "elec_hinge"
text = 'Electricity'
plot_mae(data_1, err, v1, name, text)

data_1 = [0.1342, 0.112195]
err = [0.01636, 0.000955]
v1 = min(data_1)*0.1
name = "twitter_hinge"
text = 'Twitter'
plot_mae(data_1, err, v1, name, text)

legend_plot()
