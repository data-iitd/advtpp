import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import pdb
import matplotlib.colors as mcolors

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 2, 4
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath,amsfonts} \usepackage{bm} \boldmath"

def preprocess_colors(colors):
	element = colors.pop(-1)
	colors.insert(6, element)
	return colors

def bar_plot(means, stds, name, v1, v2, btype, wd):
	texty = min(means)*0.1

	ind = np.arange(len(means))*0.12
	
	colors = list(mcolors.TABLEAU_COLORS)
	colors.pop(2)

	plt.figure()
	plt.ylim(v1, v2)
	plt.locator_params(axis='y', nbins=5)
	plt.bar(ind, means, color=colors, align='center', linewidth=1, edgecolor='k', width = 0.1)
	plt.text(0.05, v1 + 0.1*(v1+v2)/2, r'\textbf{'+wd+'}', fontsize=20, bbox=dict(facecolor='white', alpha=1))
	# plt.xticks(ind,[r'\textbf{GPT-4-Vision}', r'\textbf{LLaVA}', r'\textbf{Mini-GPT}'], fontsize=16)
	plt.xticks(ind,[r'', r'', r''], fontsize=16)

	if btype == 'mpa':
		plt.ylabel(r'\textbf{MPA} $\rightarrow$', fontsize=20)
	else:
		plt.ylabel(r'\textbf{MAE} $\rightarrow$', fontsize=20)

	plt.yticks(fontsize=20)
	plt.savefig(name+'_'+btype+".pdf", bbox_inches='tight', pad_inches = 0)

def legend_plot():
	patterns = ["\\" , "/" , "+" , "-", ".", "*","x", "o", "O"]

	colors = list(mcolors.TABLEAU_COLORS)
	colors.pop(2)

	x = 2
	y = 3
	fig = plt.figure("Line plot")
	legendFig = plt.figure("Legend plot")
	ax = fig.add_subplot(111)
	line0, = ax.bar(x, y, facecolor=colors[0], edgecolor='k')
	line1, = ax.bar(x, y, facecolor=colors[1], edgecolor='k')
	line2, = ax.bar(x, y, facecolor=colors[2], edgecolor='k')
	line3, = ax.bar(x, y, facecolor=colors[3], edgecolor='k')
	line4, = ax.bar(x, y, facecolor=colors[4], edgecolor='k')
	line5, = ax.bar(x, y, facecolor=colors[5], edgecolor='k')


	ax.legend()
	legendFig.legend([line0, line1, line2, line3, line4, line5], 
		[r'\textbf{PERM-TPP (WB)}', r'\textbf{PGD (WB)}', r'\textbf{RTS-D (WB)}'], loc='center', prop={'size': 28}, frameon=False, ncol=6)
	legendFig.savefig('legend_wd.pdf', bbox_inches='tight', pad_inches = 0.01)


col = [r'\textbf{PERM-TPP}', r'\textbf{PGD}', r'\textbf{RTS}']

wdist = [0.676, 1.036, 1.343]

# MPA
# our, pgd, rts
v1 = [20.15, 26.40, 26.49]
bar_plot(v1, [], 'taobao_wdist_v1', 10, 30, 'mpa', '0.676')
v2 = [9.94, 19.82, 23.36]
bar_plot(v1, [], 'taobao_wdist_v2', 0, 30, 'mpa', '1.036')
v3 = [10.26, 18.43, 28.19]
bar_plot(v1, [], 'taobao_wdist_v3', 10, 30, 'mpa', '1.343')


# MAE
v1 = [1.51057, 0.56918, 0.72284]
bar_plot(v1, [], 'taobao_wdist_v1', 0.2, 2, 'mae', '0.676')

v2 = [1.32652, 0.64098, 0.87572]
bar_plot(v2, [], 'taobao_wdist_v2', 0.2, 2, 'mae', '1.036')

v3 = [1.35381, 0.70801, 0.92275]
bar_plot(v3, [], 'taobao_wdist_v3', 0.2, 2, 'mae', '1.343')


legend_plot()
