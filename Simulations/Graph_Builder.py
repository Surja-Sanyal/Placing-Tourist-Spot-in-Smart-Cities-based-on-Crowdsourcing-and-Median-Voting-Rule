#Program:   Single Peaked Preference
#Inputs:    TBD
#Outputs:   TBD
#Author:    Surja Sanyal
#Date:      29 JUN 2022
#Comments:  None




##   Start of Code   ##


#   Imports    #

import os
import re
import sys
import csv
import math
import copy
import time
import json
import psutil
import shutil
import random
import datetime
import platform
import traceback
import itertools
import numpy as np
import multiprocessing
from textwrap import wrap
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr




##  Global environment   ##

#   Customize here  #
AGENTS                  = (100, 1000)                           #Agents involved. Min to Max.

#   Do not change   #
LOCK                    = multiprocessing.Lock()
DATA_LOAD_LOCATION      = os.path.dirname(sys.argv[0]) + "/"    #Local data load location
DATA_STORE_LOCATION     = os.path.dirname(sys.argv[0]) + "/"    #Local data store location
#DATA_LOAD_LOCATION     = "/content" + "/"                      #Drive data load location
#DATA_STORE_LOCATION    = "/content" + "/"                      #Drive data store location




##  Function definitions    ##


#   Print with lock    #
def print_locked(*content, sep=" ", end="\n"):

	store = os.path.dirname(sys.argv[0]) + "/"
    
	print (*content, sep = sep, end = end)

	print (*content, sep = sep, end = end, file=open(store + "/Log_Files/_Log_" +str(sys.argv[0].split("\\")[-1].split('.')[0]) + ".txt", 'a'))


#   Plot graph   #
def display_line_graph_1(store, step, precision, y_label, y, file_name):

    
        #   Set data    #
	x = [item for item in range(0, 100)]
	min_y, max_y = np.min(y), np.max(y)

	min_y = min_y - (min_y % step)
	max_y = max_y + step/2.
	
	colors = ['r', 'g']   #['k', 'r', 'b', 'g']
	labels = ['Mean', 'Median']      #['Default', 'SPP-RR', 'SPP-AR', 'SPP-MR']
	

	#   Create the figure   #
	comparison = plt.figure(file_name)
	ax = plt.subplot(111, xlim=(0, 100))#, ylim=(min_y, max_y))

	#ax.axhline(y=1, color='k', linestyle=':')
	[ax.plot(x, y[method], label=labels[method], color=colors[method], linestyle = '-') for method in range(len(y))]

	
	#   Customize plot   #
	ax.set_xticks([tick for tick in range(0, 100 + 1, 10)])
	ax.set_xticklabels([str(tick) for tick in range(0, 100 + 1, 10)])

	#print([round(tick, precision) for tick in np.arange(min_y, max_y + step/2, step)], )
	#ax.set_yticks([round(tick, precision) for tick in np.arange(min_y, max_y + step/2, step)])
	#ax.set_yticklabels([format(round(tick, precision), "." + str(precision) + "f") for tick in np.arange(min_y, max_y + step/2, step)])
	
	plt.ylabel(y_label + r'$\rightarrow$', fontsize=15)
	plt.xlabel('\nAgent ID ' + r'$\rightarrow$', fontsize=15, labelpad=-12)
	
	plt.grid(axis = 'y')
	
	ax.legend(title='Methods', title_fontsize=15, fontsize=10, ncol=4, loc='best')
    

	#   Save plot   #
	comparison.savefig(store + "/Graphs/" + file_name + ".pdf", bbox_inches='tight')


	#   Clean plot    #
	plt.clf()


#   Plot graph   #
def display_line_graph_2(store, step, precision, average, median, file_name):

    
        #   Set data    #
	x = [item for item in range(AGENTS[0], AGENTS[1] + 1, 100)]
	min_y, max_y = round(min([min(method) for method in average + median]), precision), \
                       round(max([max(method) for method in average + median]), precision)

	average, median = [[((value - min_y) / (max_y - min_y)) * (1 - 0) + 0 for value in row] for row in average], \
                          [[((value - min_y) / (max_y - min_y)) * (1 - 0) + 0 for value in row] for row in median]
	
	min_y, max_y = 0.0, 1.2
	
	colors = ['b', 'y', 'r', 'g']
	labels_1 = ['Mean: ', 'Median: ']
	labels_2 = ['25', '50', '75', '100']
	linestyle = [':', '--']
	

	#   Create the figure   #
	comparison = plt.figure(file_name)
	ax = plt.subplot(111, xlim=(AGENTS[0], AGENTS[1]), ylim=(min_y, max_y))

	#ax.axhline(y=1, color='k', linestyle=':')
	[ax.plot(x, average[level], label=labels_1[0] + labels_2[level] + " %", color=colors[level], linestyle = linestyle[0]) \
         for level in range(len(average))]
	[ax.plot(x, median[level], label=labels_1[1] + labels_2[level] + " %", color=colors[level], linestyle = linestyle[1]) \
         for level in range(len(median))]

	
	#   Customize plot   #
	ax.set_xticks([tick for tick in range(AGENTS[0], AGENTS[1] + 1, 100)])
	ax.set_xticklabels([str(tick) for tick in range(AGENTS[0], AGENTS[1] + 1, 100)])
	ax.set_yticks([round(tick, precision) for tick in np.arange(min_y, max_y, step)])
	ax.set_yticklabels([format(round(tick, precision), "." + str(precision) + "f") for tick in np.arange(min_y, max_y, step)])
	
	plt.ylabel('Normalized Utility Deviation ' + r'$\rightarrow$', fontsize=15)
	plt.xlabel('\nTotal Agent Count ' + r'$\rightarrow$', fontsize=15, labelpad=-12)
	
	plt.grid(axis = 'y')
	
	ax.legend(title='Methods: Misreport %', title_fontsize=15, fontsize=10, ncol=1, loc='best')
    

	#   Save plot   #
	comparison.savefig(store + "/Graphs/" + file_name + ".pdf", bbox_inches='tight')


	#   Clean plot    #
	plt.clf()



#   Plot graph   #
def display_line_graph_3(store, step, precision, data, a_pearson, m_pearson, a_spearman, m_spearman, file_name):

    
        #   Set data    #
	x, y = data[0], data[1:]
	y = [[-value for value in row] for row in y]
	min_y, max_y = round(min([min(method) for method in y]), precision), \
                       round(max([max(method) for method in y]), precision)

	average, median = y[0], y[1]
	
	min_y, max_y = 0.0, 1.2
	
	colors = ['r', 'g']
	labels_1 = ['Mean: ', 'Median: ']
	linestyle = [':', '--', '-', '-']
	

	#   Create the figure   #
	comparison = plt.figure(file_name)
	ax = plt.subplot(111)#, ylim=(min_y, max_y))

	ax.axhline(y=0, color='k', linestyle=':')
	ax.plot(x, average, label=labels_1[0] + format(a_pearson, ".4f") + ", " + format(a_spearman, ".4f"), color=colors[0], linestyle = linestyle[2])
	ax.plot(x, median, label=labels_1[1] + format(m_pearson, ".4f") + ", " + format(m_spearman, ".4f"), color=colors[1], linestyle = linestyle[3])
	#ax.plot(sorted(x), [val for _, val in sorted(zip(x, average))], label=labels_1[0] + format(a_pearson, ".4f") + ", " + format(a_spearmanr, ".4f"), color=colors[0], linestyle = linestyle[0])
	#ax.plot(sorted(x), [val for _, val in sorted(zip(x, median))], label=labels_1[1] + format(m_pearson, ".4f") + ", " + format(m_spearmanr, ".4f")

	
	#   Customize plot   #
	#plt.xticks([])
	#plt.yticks([])
	#ax.set_xticks([round(tick, precision) for tick in np.arange(0, 1.1, 0.2)])
	#ax.set_xticklabels([format(round(tick, precision), "." + str(precision) + "f") for tick in np.arange(0, 1.1, 0.2)])
	#ax.set_yticks([round(-tick, precision) for tick in np.arange(min_y, max_y, step)])
	#ax.set_yticklabels([format(round(-tick, precision), "." + str(precision) + "f") for tick in np.arange(min_y, max_y, step)])
	
	plt.ylabel('Agent Change in Utility ' + r'$\rightarrow$', fontsize=15)
	plt.xlabel('\nAgent Preference Deviation ' + r'$\rightarrow$', fontsize=15, labelpad=-12)
	
	plt.grid(axis = 'y')
	
	ax.legend(title="Methods with\nPearson's Corr., &\nSpearman's corr.", title_fontsize=15, fontsize=10, ncol=1, loc='best')
    

	#   Save plot   #
	comparison.savefig(store + "/Graphs/" + file_name + ".pdf", bbox_inches='tight')


	#   Clean plot    #
	plt.clf()






##  The main function   ##

#   Main    #
def main():


    #   Global settings #
    load, store = DATA_LOAD_LOCATION, DATA_STORE_LOCATION


    #   Get graph data  #
    h_data_1D = json.load(open(store + "/Statistics/honest_1D.json"))
    h_data_2D = json.load(open(store + "/Statistics/honest_2D.json"))
    d_data_1D = json.load(open(store + "/Statistics/dishonest_1D.json"))
    d_data_2D = json.load(open(store + "/Statistics/dishonest_2D.json"))
    d_data_1D_2D = json.load(open(store + "/Statistics/dishonest_stats_1D_2D.json"))
    d_u_1D_2D = json.load(open(store + "/Statistics/deviation_utility_1D_2D.json"))


    #   Display graphs  #
    display_line_graph_1(store, 0.1, 1, 'Agent Utility ', h_data_1D, "Honest_Utility_1D")
    display_line_graph_1(store, 0.1, 1, 'Agent Utility ', h_data_2D, "Honest_Utility_2D")
    display_line_graph_1(store, 0.1, 1, 'Agent Change in Utility ', d_data_1D, "Dishonest_Utility_1D")
    display_line_graph_1(store, 0.1, 1, 'Agent Change in Utility ', d_data_2D, "Dishonest_Utility_2D")



    y = [json.loads(item) for item in d_data_1D_2D]
    
    average, median = [[row[0] for row in level] for level in y[0]], [[row[1] for row in level] for level in y[0]]
    average, median = list(map(list, zip(*average))), list(map(list, zip(*median)))

    display_line_graph_2(store, 0.2, 1, average, median, "Dishonest_Variable_Utility_1D")

    average, median = [[row[0] for row in level] for level in y[1]], [[row[1] for row in level] for level in y[1]]
    average, median = list(map(list, zip(*average))), list(map(list, zip(*median)))
    
    display_line_graph_2(store, 0.2, 1, average, median, "Dishonest_Variable_Utility_2D")



    data_1D, data_2D = d_u_1D_2D[:3], d_u_1D_2D[3:]
    #print(data_1D, data_2D, sep="\n\n")

    a_pearson_1D, _ = pearsonr(data_1D[0], data_1D[1])
    m_pearson_1D, _ = pearsonr(data_1D[0], data_1D[2])
    a_spearman_1D, _ = spearmanr(data_1D[0], data_1D[1])
    m_spearman_1D, _ = spearmanr(data_1D[0], data_1D[2])

    a_pearson_2D, _ = pearsonr(data_2D[0], data_2D[1])
    m_pearson_2D, _ = pearsonr(data_2D[0], data_2D[2])
    a_spearman_2D, _ = spearmanr(data_2D[0], data_2D[1])
    m_spearman_2D, _ = spearmanr(data_2D[0], data_2D[2])

    display_line_graph_3(store, 0.2, 1, data_1D, a_pearson_1D, m_pearson_1D, a_spearman_1D, m_spearman_1D, "Deviation_vs_Utility_1D")
    display_line_graph_3(store, 0.2, 1, data_2D, a_pearson_2D, m_pearson_2D, a_spearman_2D, m_spearman_2D, "Deviation_vs_Utility_2D")

    print_locked("Graph generation complete!")

    




##  Call the main function  ##

#   Initiation  #
if __name__=="__main__":

    try:

        #   Start logging to file     #        
        print_locked('\n\n\n\n{:.{align}{width}}'.format("Execution Start at: " 
            + str(datetime.datetime.now()), align='<', width=150), end="\n\n")

        print_locked("\n\nPython Version:\n\n" + str(platform.python_version()))
        
        print_locked("\n\nProgram Name:\n\n" + str(sys.argv[0].split("\\")[-1]))
        
        print_locked("\n\nProgram Path:\n\n" + os.path.dirname(sys.argv[0]))
        
        print_locked("\n\nProgram Name With Path:\n\n" + str(sys.argv[0]))

        print_locked("\n\nProgram arguments aubmitted:\n\n" + str(sys.argv[1:]))

        print_locked('\n\n\n\n{:.{align}{width}}'.format("Program Body Start:", align='<', width=50), end="\n\n\n")

        
        #   Clear the terminal  #
        #os.system("clear")

        
        #   Initiate lock object    #
        #lock = multiprocessing.Lock()


        #   Initiate pool objects   #
        #pool = multiprocessing.Pool(multiprocessing.cpu_count())

        
        #   Call the main program   #
        start = datetime.datetime.now()

        
        #    main()
        if (len(sys.argv) > 1):
                main(int(sys.argv[1]), int(sys.argv[2]))
        else:
                main()


        print_locked('\n\n{:.{align}{width}}'.format("Program Body End:", align='<', width=50), end="\n\n\n\n")
        
        print_locked("\nProgram execution time:\t\t", datetime.datetime.now() - start, "hours\n")


        #    Wait for manual exit
        #input('\n\n\n\nPress ENTER to exit: ')

        
        #   Close Pool object    #
        #pool.close()


    except Exception:
    
        print_locked(traceback.format_exc())


##   End of Code   ##

