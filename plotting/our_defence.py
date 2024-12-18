import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import pdb
from matplotlib import colormaps
import matplotlib.colors as mcolors

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 5, 3
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath,amsfonts} \usepackage{bm} \boldmath"

def bar_plot(data_1, data_2, col, name, v1, v2, restype):
	data1 = pd.DataFrame(data_1, columns=col).assign(Location=1)
	data2 = pd.DataFrame(data_2, columns=col).assign(Location=2)
	# data3 = pd.DataFrame(data_3, columns=col).assign(Location=3)
	# patterns = ["\\\\", "\\\\", "\\\\",  "//", "//", "//",  "++", "++", "++",  "--", "--", "--",  "..", "..", "..",  "*", "*", "*",  "xx", "xx", "xx",  "oo", "oo", "oo",  "OO", "OO", "OO"]

	cdf = pd.concat([data1, data2])
	mdf = pd.melt(cdf, id_vars=['Location'], var_name=['Letter'])

	# colors = ["salmon", 'cornflowerblue', 'seagreen', "#56B4E9", "#D55E00", "#CC79A7",  "sandybrown", "#E69F00"]
	colors = list(mcolors.TABLEAU_COLORS)
	colors.pop(4)
	colors.pop(4)
	colors.pop(4)
	
	# ax = sns.barplot(x="Location", y="value", hatch=patterns, hue="Letter", data=mdf, errwidth=3, errcolor='k', palette=colors, edgecolor='k')
	ax = sns.barplot(x="Location", y="value", hue="Letter", data=mdf, errwidth=0, errcolor='k', palette=colors, edgecolor='k', capsize=0.01)
	ax.set(ylim=(v1, v2))

	ax.yaxis.set_major_locator(plt.MaxNLocator(5))
	ax.patch.set_edgecolor('black')  
	ax.patch.set_linewidth('1.5')

	ax.set_xlabel(r'', fontsize=24)
	if restype == 'mpa':
		ax.set_ylabel(r'\textbf{MPA}$\rightarrow$', fontsize=24)
	else:
		ax.set_ylabel(r'\textbf{MAE}$\rightarrow$', fontsize=24)

	ax.set_xticklabels([r'\textbf{BB}', r'\textbf{WB}'], fontsize=20)
	
	ax.tick_params(axis='y', which='major', pad=-1)
	ax.tick_params(axis='x', which='major', pad=-1)
	plt.yticks(fontsize=20)

	# ax.legend(loc='lower center', ncol=3, fancybox=True, prop={'size': 16})
	ax.get_legend().remove()
	plt.tight_layout()
	plt.margins(0.02,0.02)

	plt.savefig(name+"_"+restype+".pdf", bbox_inches='tight', pad_inches = 0)
	plt.close()

def legend_plot():
	patterns = ["\\" , "/" , "+" , "-", ".", "*","x", "o", "O"]
	colors = list(mcolors.TABLEAU_COLORS)
	colors.pop(4)
	colors.pop(4)
	colors.pop(4)
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
	line6, = ax.bar(x, y, facecolor=colors[6], edgecolor='k')

	ax.legend()
	legendFig.legend([line0, line1, line2, line3, line4, line5, line6], 
		[r'\textbf{PERM-TPP}', r'\textbf{PGD}', r'\textbf{MIFGSM}', r'\textbf{RTS-D}', r'\textbf{RTS-P}'], loc='center', prop={'size': 28}, frameon=False, ncol=6)
	legendFig.savefig('legend_def.pdf', bbox_inches='tight', pad_inches = 0.01)

col = [r'\textbf{PERM-TPP}', r'\textbf{PGD}', r'\textbf{MIFGSM}', r'\textbf{RTS-D}', r'\textbf{RTS-P}']

# TAOBAO
name = 'taobao'
# FOR MPA
bb = [[46.99, 41.52, 41.42, 41.58, 45.76], [46.99, 41.52, 41.42, 41.58, 45.76]]
wb = [[46.61, 18.56, 19.94, 21.12, 40.78], [46.61, 18.56, 19.94, 21.12, 40.78]]
bar_plot(bb, wb, col, name, 10, 50, 'mpa')

# FOR MAE
bb = [[0.45841, 1.16311, 1.13842, 1.06107, 0.49881], [0.45841, 1.16311, 1.13842, 1.06107, 0.49881]]
wb = [[0.52561, 1.26513, 1.29521, 1.22826, 0.51301], [0.52561, 1.26513, 1.29521, 1.22826, 0.51301]]
bar_plot(bb, wb, col, name, 0.4, 1.4, 'mae')



# HEALTH
name = 'health'
# FOR MPA
bb = [[63.10, 49.83, 61.65, 60.29, 62.30], [63.10, 49.83, 61.65, 60.29, 62.30]]
wb = [[63.10, 40.80, 53.67, 49.14, 58.46], [63.10, 40.80, 53.67, 49.14, 58.46]]
bar_plot(bb, wb, col, name, 30, 70, 'mpa')

# FOR MAE
bb = [[0.03956, 0.04925, 0.03493, 0.03513, 0.03379], [0.03956, 0.04925, 0.03493, 0.03513, 0.03379]]
wb = [[0.03956, 0.04233, 0.04876, 0.04184, 0.03378], [0.03956, 0.04233, 0.04876, 0.04184, 0.03378]]
bar_plot(bb, wb, col, name, 0.02, 0.05, 'mae')


legend_plot()
