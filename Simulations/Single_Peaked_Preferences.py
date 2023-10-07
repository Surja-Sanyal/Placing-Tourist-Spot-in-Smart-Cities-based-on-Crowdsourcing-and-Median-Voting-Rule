#Program:   Single Peaked Preference
#Inputs:    None
#Outputs:   Files: (Appends)
#           1. DATA_STORE_LOCATION/Statistics/Per Round.txt
#           2. DATA_STORE_LOCATION/Statistics/Total Cumulative.txt
#           3. DATA_STORE_LOCATION/Statistics/Individual Cumulative.txt
#Author:    Surja Sanyal
#Email:     hi.surja06@gmail.com
#Date:      02 JUL 2022
#Comments:  1. Please create a folder named "Statistics" in the location DATA_STORE_LOCATION. Outputs will be saved there.




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
from scipy.stats import truncnorm




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



#   Deviation vs Utility        #
def dev_util(store, location_list, agents):


        #   Prepare data   #
        pivot = int(agents/4)
        
        x_1d, x_2d, y_2d = sorted(location_list)[:agents], \
                           sorted(location_list)[:agents], \
                           sorted(location_list)[:agents]


	#   Misreports   #
        m_x_1d, m_x_2d, m_y_2d = sorted(location_list)[-agents:], sorted(location_list)[-agents:], sorted(location_list)[-agents:]
        

	#   Generate 1D stats   #
        random_1D, average_1D, median_1D = random.choice(x_1d), np.mean(x_1d), np.median(x_1d)


	#   Generate 2D stats   #
        random_x, average_x, median_x = random.choice(x_2d), np.mean(x_2d), np.median(x_2d)
        random_y, average_y, median_y = random.choice(y_2d), np.mean(y_2d), np.median(y_2d)


	#   Honest 1D stats    #
        h_utility_1D_random = np.array([1 - abs(random_1D - x_1d[pivot]) for agent in range(agents)])
        h_utility_1D_average = np.array([1 - abs(average_1D - x_1d[pivot]) for agent in range(agents)])
        h_utility_1D_median = np.array([1 - abs(median_1D - x_1d[pivot]) for agent in range(agents)])


	#    Honest 2D stats    #
        h_utility_2D_random = np.array([1 - math.sqrt((random_x - x_2d[pivot])**2 + (random_y - y_2d[pivot])**2) \
                                      / math.sqrt(1**2 + 1**2) for agent in range(agents)])
        h_utility_2D_average = np.array([1 - math.sqrt((average_x - x_2d[pivot])**2 + (average_y - y_2d[pivot])**2) \
                                      / math.sqrt(1**2 + 1**2) for agent in range(agents)])
        h_utility_2D_median = np.array([1 - math.sqrt((median_x - x_2d[pivot])**2 + (median_y - y_2d[pivot])**2) \
                                      / math.sqrt(1**2 + 1**2) for agent in range(agents)])


        #   Misreporting 1D stats   #
        deviation_1D = np.array([abs(x_1d[pivot] - x_1d[agent]) for agent in range(agents)])
        m_utility_1D_random, m_utility_1D_average, m_utility_1D_median = np.ones(agents), np.ones(agents), np.ones(agents)
	
        for agent in range(agents):

                m_random_x, m_average_x, m_median_x = random.choice([x_1d[agent]] + [x for i, x in enumerate(x_1d) if i != pivot]), \
						      np.mean([x_1d[agent]] + [x for i, x in enumerate(x_1d) if i != pivot]), \
                                                      np.median([x_1d[agent]] + [x for i, x in enumerate(x_1d) if i != pivot])

                m_utility_1D_random[agent] -= abs(m_random_x - x_1d[pivot])
                m_utility_1D_average[agent] -= abs(m_average_x - x_1d[pivot])
                m_utility_1D_median[agent] -= abs(m_median_x - x_1d[pivot])


	#   Misreporting 2D stats   #
        deviation_2D = np.array([math.sqrt(abs(x_2d[pivot] - x_2d[agent])**2 + abs(y_2d[pivot] - y_2d[agent])**2) for agent in range(agents)])
        m_utility_2D_random, m_utility_2D_average, m_utility_2D_median = np.ones(agents), np.ones(agents), np.ones(agents)

        for agent in range(agents):

                m_random_x, m_average_x, m_median_x = random.choice([x_2d[agent]] + [x for i, x in enumerate(x_2d) if i != pivot]), \
						      np.mean([x_2d[agent]] + [x for i, x in enumerate(x_2d) if i != pivot]), \
                                                      np.median([x_2d[agent]] + [x for i, x in enumerate(x_2d) if i != pivot])
		
                m_random_y, m_average_y, m_median_y = random.choice([y_2d[agent]] + [x for i, x in enumerate(y_2d) if i != pivot]), \
						      np.mean([y_2d[agent]] + [x for i, x in enumerate(y_2d) if i != pivot]), \
                                                      np.median([y_2d[agent]] + [x for i, x in enumerate(y_2d) if i != pivot])

                m_utility_2D_random[agent] -= (math.sqrt((m_random_x - x_2d[pivot])**2 + (m_random_y - y_2d[pivot])**2)) / math.sqrt(1**2 + 1**2)
		
                m_utility_2D_average[agent] -= (math.sqrt((m_average_x - x_2d[pivot])**2 + (m_average_y - y_2d[pivot])**2)) / math.sqrt(1**2 + 1**2)
		
                m_utility_2D_median[agent] -= (math.sqrt((m_median_x - x_2d[pivot])**2 + (m_median_y - y_2d[pivot])**2)) / math.sqrt(1**2 + 1**2)


	#   Deviation in utility   #
        m_utility_1D_random = np.array([(honest - dishonest) for dishonest, honest in zip(m_utility_1D_random, h_utility_1D_random)])
        m_utility_1D_average = np.array([(honest - dishonest) for dishonest, honest in zip(m_utility_1D_average, h_utility_1D_average)])
        m_utility_1D_median = np.array([(honest - dishonest) for dishonest, honest in zip(m_utility_1D_median, h_utility_1D_median)])

        m_utility_2D_random = np.array([(honest - dishonest) for dishonest, honest in zip(m_utility_2D_random, h_utility_2D_random)])
        m_utility_2D_average = np.array([(honest - dishonest) for dishonest, honest in zip(m_utility_2D_average, h_utility_2D_average)])
        m_utility_2D_median = np.array([(honest - dishonest) for dishonest, honest in zip(m_utility_2D_median, h_utility_2D_median)])


	#   Save data   #
        json.dump((deviation_1D.tolist(), m_utility_1D_average.tolist(), m_utility_1D_median.tolist(), \
                   deviation_2D.tolist(), m_utility_2D_average.tolist(), m_utility_2D_median.tolist()), \
                  open(store + "/Statistics/deviation_utility_1D_2D.json", "w"), indent=2)




#   Driver code for different agents    #
def driver_code(load, store, agents):


        #   Announce current execution  #
	print_locked("\nExecuting for agents =", agents)
        

        #   Read JSON file   #
	location_list = json.load(open(store + "_location_list.json"))


	#   Generate 1D locations   #
	x_1d = list(random.choices(location_list, k=agents))


	#   Generate 2D locations   #
	x_2d, y_2d = list(random.choices(location_list, k=agents)), list(random.choices(location_list, k=agents))


	#   Generate 1D stats   #
	random_1D, average_1D, median_1D = random.choice(x_1d), np.mean(x_1d), np.median(x_1d)


	#   Generate 2D stats   #
	random_x, average_x, median_x = random.choice(x_2d), np.mean(x_2d), np.median(x_2d)
	random_y, average_y, median_y = random.choice(y_2d), np.mean(y_2d), np.median(y_2d)


	#   Honest 1D stats    #
	h_utility_1D_random = np.array([1 - abs(random_1D - x_1d[agent]) for agent in range(agents)])
	h_utility_1D_average = np.array([1 - abs(average_1D - x_1d[agent]) for agent in range(agents)])
	h_utility_1D_median = np.array([1 - abs(median_1D - x_1d[agent]) for agent in range(agents)])


	#    Honest 2D stats    #
	h_utility_2D_random = np.array([1 - math.sqrt((random_x - x_2d[agent])**2 + (random_y - y_2d[agent])**2) \
                                      / math.sqrt(1**2 + 1**2) for agent in range(agents)])
	h_utility_2D_average = np.array([1 - math.sqrt((average_x - x_2d[agent])**2 + (average_y - y_2d[agent])**2) \
                                      / math.sqrt(1**2 + 1**2) for agent in range(agents)])
	h_utility_2D_median = np.array([1 - math.sqrt((median_x - x_2d[agent])**2 + (median_y - y_2d[agent])**2) \
                                      / math.sqrt(1**2 + 1**2) for agent in range(agents)])


        #   Misreporting 1D stats   #
	m_x_1d = list(random.choices(location_list, k=agents))
	
	m_utility_1D_random, m_utility_1D_average, m_utility_1D_median = np.ones(agents), np.ones(agents), np.ones(agents)

	for agent in range(agents):

		m_random_x, m_average_x, m_median_x = random.choice([m_x_1d[agent]] + [x for i, x in enumerate(x_1d) if i != agent]), \
						      np.mean([m_x_1d[agent]] + [x for i, x in enumerate(x_1d) if i != agent]), \
                                                      np.median([m_x_1d[agent]] + [x for i, x in enumerate(x_1d) if i != agent])

		m_utility_1D_random[agent] -= abs(m_random_x - x_1d[agent])
		m_utility_1D_average[agent] -= abs(m_average_x - x_1d[agent])
		m_utility_1D_median[agent] -= abs(m_median_x - x_1d[agent])


	for portion in (0.25, 0.50, 0.75, 1.):

		m_random_x, m_average_x, m_median_x = random.choice(m_x_1d[:int(portion * agents)] + x_1d[int(portion * agents):]), \
						      np.mean(m_x_1d[:int(portion * agents)] + x_1d[int(portion * agents):]), \
                                                      np.median(m_x_1d[:int(portion * agents)] + x_1d[int(portion * agents):])

		p_utility_1D_random = sum([abs(abs(random_1D - x_1d[agent]) - abs(m_random_x - x_1d[agent])) for agent in range(agents)])
		p_utility_1D_average = sum([abs(abs(average_1D - x_1d[agent]) - abs(m_average_x - x_1d[agent])) for agent in range(agents)])
		p_utility_1D_median = sum([abs(abs(median_1D - x_1d[agent]) - abs(m_median_x - x_1d[agent])) for agent in range(agents)])

		percentage = int(portion * 100)
			
		json.dump((p_utility_1D_average, p_utility_1D_median), \
			open(store + "/Statistics/dishonest_1D_" + str(agents) + "_" + str(percentage) + ".json", "w"), indent=2)

	


	#   Misreporting 2D stats   #
	m_x_2d, m_y_2d = list(random.choices(location_list, k=agents)), list(random.choices(location_list, k=agents))
	
	m_utility_2D_random, m_utility_2D_average, m_utility_2D_median = np.ones(agents), np.ones(agents), np.ones(agents)

	for agent in range(agents):

		m_random_x, m_average_x, m_median_x = random.choice([m_x_2d[agent]] + [x for i, x in enumerate(x_2d) if i != agent]), \
						      np.mean([m_x_2d[agent]] + [x for i, x in enumerate(x_2d) if i != agent]), \
                                                      np.median([m_x_2d[agent]] + [x for i, x in enumerate(x_2d) if i != agent])
		
		m_random_y, m_average_y, m_median_y = random.choice([m_y_2d[agent]] + [x for i, x in enumerate(y_2d) if i != agent]), \
						      np.mean([m_y_2d[agent]] + [x for i, x in enumerate(y_2d) if i != agent]), \
                                                      np.median([m_y_2d[agent]] + [x for i, x in enumerate(y_2d) if i != agent])

		m_utility_2D_random[agent] -= (math.sqrt((m_random_x - x_2d[agent])**2 + (m_random_y - y_2d[agent])**2)) / math.sqrt(1**2 + 1**2)
		
		m_utility_2D_average[agent] -= (math.sqrt((m_average_x - x_2d[agent])**2 + (m_average_y - y_2d[agent])**2)) / math.sqrt(1**2 + 1**2)
		
		m_utility_2D_median[agent] -= (math.sqrt((m_median_x - x_2d[agent])**2 + (m_median_y - y_2d[agent])**2)) / math.sqrt(1**2 + 1**2)


	for portion in (0.25, 0.50, 0.75, 1.):

		m_random_x, m_average_x, m_median_x = random.choice(m_x_2d[:int(portion * agents)] + x_2d[int(portion * agents):]), \
						      np.mean(m_x_2d[:int(portion * agents)] + x_2d[int(portion * agents):]), \
                                                      np.median(m_x_2d[:int(portion * agents)] + x_2d[int(portion * agents):])
		
		m_random_y, m_average_y, m_median_y = random.choice(m_y_2d[:int(portion * agents)] + y_2d[int(portion * agents):]), \
						      np.mean(m_y_2d[:int(portion * agents)] + y_2d[int(portion * agents):]), \
                                                      np.median(m_y_2d[:int(portion * agents)] + y_2d[int(portion * agents):])

		p_utility_2D_random = sum([abs((math.sqrt((random_x - x_2d[agent])**2 + (random_y - y_2d[agent])**2) \
                                            - math.sqrt((m_random_x - x_2d[agent])**2 + (m_random_y - y_2d[agent])**2)) \
                                              / math.sqrt(1**2 + 1**2)) for agent in range(agents)])
		
		p_utility_2D_average = sum([abs((math.sqrt((average_x - x_2d[agent])**2 + (average_y - y_2d[agent])**2) \
                                            - math.sqrt((m_average_x - x_2d[agent])**2 + (m_average_y - y_2d[agent])**2)) \
                                              / math.sqrt(1**2 + 1**2)) for agent in range(agents)])
		
		p_utility_2D_median = sum([abs((math.sqrt((median_x - x_2d[agent])**2 + (median_y - y_2d[agent])**2) \
                                            - math.sqrt((m_median_x - x_2d[agent])**2 + (m_median_y - y_2d[agent])**2)) \
                                              / math.sqrt(1**2 + 1**2)) for agent in range(agents)])

		percentage = int(portion * 100)
			
		json.dump((p_utility_2D_average, p_utility_2D_median), \
			open(store + "/Statistics/dishonest_2D_" + str(agents) + "_" + str(percentage) + ".json", "w"), indent=2)


	#   Deviation in utility   #
	m_utility_1D_random = np.array([(dishonest - honest) for dishonest, honest in zip(m_utility_1D_random, h_utility_1D_random)])
	m_utility_1D_average = np.array([(dishonest - honest) for dishonest, honest in zip(m_utility_1D_average, h_utility_1D_average)])
	m_utility_1D_median = np.array([(dishonest - honest) for dishonest, honest in zip(m_utility_1D_median, h_utility_1D_median)])

	m_utility_2D_random = np.array([(dishonest - honest) for dishonest, honest in zip(m_utility_2D_random, h_utility_2D_random)])
	m_utility_2D_average = np.array([(dishonest - honest) for dishonest, honest in zip(m_utility_2D_average, h_utility_2D_average)])
	m_utility_2D_median = np.array([(dishonest - honest) for dishonest, honest in zip(m_utility_2D_median, h_utility_2D_median)])


	#   Only for agents == 100   #
	if agents == 100:

                #   Deviation vs Utility   #
		dev_util(store, location_list, agents)

                
                #   Save data   #
		json.dump((h_utility_1D_average.tolist(), h_utility_1D_median.tolist()), \
			open(store + "/Statistics/honest_1D.json", "w"), indent=2)

		json.dump((h_utility_2D_average.tolist(), h_utility_2D_median.tolist()), \
			open(store + "/Statistics/honest_2D.json", "w"), indent=2)
	
		json.dump((m_utility_1D_average.tolist(), m_utility_1D_median.tolist()), \
			open(store + "/Statistics/dishonest_1D.json", "w"), indent=2)

		json.dump((m_utility_2D_average.tolist(), m_utility_2D_median.tolist()), \
			open(store + "/Statistics/dishonest_2D.json", "w"), indent=2)



#   Collect data        #
def collect_data(load, store, start, end, step):


        data_1D, data_2D = [], []
        for agents in range(start, end, step):

                execution_1D, execution_2D = [], []
                for agent in (int(agents / 4), int(agents / 2), int(3 * agents / 4), agents - 1):

                        percentage = 25 if agent == int(agents / 4) \
                                        else (50 if agent == int(agents / 2) else (75 if agent == int(3 * agents / 4) else 100))

                        execution_1D.append(json.load(open(store + "/Statistics/dishonest_1D_" + str(agents) + "_" + str(percentage) + ".json")))
                        execution_2D.append(json.load(open(store + "/Statistics/dishonest_2D_" + str(agents) + "_" + str(percentage) + ".json")))

                        os.remove(store + "/Statistics/dishonest_1D_" + str(agents) + "_" + str(percentage) + ".json")
                        os.remove(store + "/Statistics/dishonest_2D_" + str(agents) + "_" + str(percentage) + ".json")

                data_1D.append(execution_1D)
                data_2D.append(execution_2D)

        json.dump((json.dumps(data_1D), json.dumps(data_2D)), open(store + "/Statistics/dishonest_stats_1D_2D.json", "w"), indent=2)





##  The main function   ##

#   Main    #
def main():


	#   Global settings	#
	load, store = DATA_LOAD_LOCATION, DATA_STORE_LOCATION
	random.seed(1)
	

	#   Call driver code    #
	[driver_code(load, store, agents) for agents in range(AGENTS[0], AGENTS[1] + 1, 100)]


	#   Collect data        #
	collect_data(load, store, AGENTS[0], AGENTS[1] + 1, 100)
	




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

