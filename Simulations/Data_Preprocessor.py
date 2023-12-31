#Program:   Data Preprocessing
#Inputs:    TBD
#Outputs:   TBD
#Author:    Surja Sanyal
#Date:      28 JUN 2022
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
from scipy.stats import truncnorm




##  Global environment   ##

#   Customize here  #
ITEMS                                   = 20                                   #Items for voting
MAX_PRICE                               = 500000                                #Max price for one item
MIN_PRICE                               = 50000                                #Min price for one item
VOLUNTEERS				= "16X"					#1X/2X/4X/8X/16X/32X				                Volunteer availability (1/2/4/8/16/32) times donors
RECEIVERS				= "4X"					#1X/2X/4X/8X/16X/32X				                Receiver availability (1/2/4/8/16/32) times donors
AGENTS					= 200					#Number								Default number of agent requests
PAYLOAD_MAX				= 100					#Number								Volunteer payload limit in kilograms
COORDINATE_MAX       	                = 500					#Number								City start (at 0) to end limit in kilometers
DAY_MAX			      	        = 18					#Number								Day start (at 0) to end limit in hours
SAVE					= "ON"					#ON/OFF								Save data
SORTING					= "END"					#START/END							Receiver sorting
PREFERENCE				= "ELIGIBLE"			        #ORIGINAL/ELIGIBLE/UPDATED			                Usage of preference lists
MANIPULATION			        = "OFF"					#ON/OFF								Manipulation of preferences

#	Thresholds	#
To		= 0.25									        #Overlap time (hours)
Tl		= 5										#Off-routing (percentage)
Tm		= 1										#Meal size  (kilograms)
Ta		= 20									        #Extra payload (percentage)
Tpm		= 20									        #Default perishable food travel distance (kilometers)
Tpnm	        = 5										#Default perishable food travel distance (kilometers)
Tnp		= 100									        #Default non-perishable food travel distance (kilometers)
Td		= 2										#Process advance start threshold for donors (hours)
Tr		= 3										#Process advance start threshold for receivers (hours)
Tw		= 10									        #Match acceptance window (minutes)

#   Do not change   #
LOCK                    = multiprocessing.Lock()								#Multiprocessing lock
CPU_COUNT               = multiprocessing.cpu_count()							        #Logical CPUs
MEMORY                  = math.ceil(psutil.virtual_memory().total/(1024.**3))	                                #RAM capacity
DATA_LOAD_LOCATION	= os.path.dirname(sys.argv[0]) + "/"					                #Data load location
DATA_STORE_LOCATION	= os.path.dirname(sys.argv[0]) + "/"				                        #Data store location
#DATA_LOAD_LOCATION	= "/content" + "/"									#Data load location
#DATA_STORE_LOCATION	= "/content" + "/"									#Data store location




##  Function definitions    ##



#   Print with lock    #
def print_locked(*content, sep=" ", end="\n"):

	store = os.path.dirname(sys.argv[0]) + "/"
    
	with open(store + "/Log_Files/_Log_" +str(sys.argv[0].split("\\")[-1].split('.')[0]) + ".txt", 'a') as log_file:
	
		try:
	    
			with lock:
	        
				print (*content, sep = sep, end = end)
				print (*content, sep = sep, end = end, file=log_file)

		except Exception:
	    
			with LOCK:
	        
				print (*content, sep = sep, end = end)
				print (*content, sep = sep, end = end, file=log_file)


#   Read CSV file   #
def read_csv():

    load, store, cmax, items, max_price, min_price = DATA_LOAD_LOCATION, DATA_STORE_LOCATION, COORDINATE_MAX, ITEMS, MAX_PRICE, MIN_PRICE
    locations = []
    
    with open(load + "Real Life Dataset/" + "Cadastral positions according to the charge's use of the city of Barcelona.csv", "r") as f:
        
        reader = csv.reader(f, delimiter=",")
        
        for i, line in enumerate(reader):
            
            if(i > 0):

                num = int("".join(re.findall(r'[\d-]',line[-1])))
                locations = locations + [num]

    max_location = max(locations)
    location_list = np.unique([round(location / max_location, 4) for location in locations])

    print_locked("Unique values generated:", len(location_list))

    #   Write data  #
    json.dump(location_list.tolist(), open(store + "_location_list.json", "w"), indent=2)





##  The main function   ##

#   Main    #
def main():
	
	#	Get volunteer settings	#
	save_setting, v_setting, r_setting, pref_setting, sort_setting, manip_setting = SAVE, VOLUNTEERS, RECEIVERS, PREFERENCE, SORTING, MANIPULATION

        #   Read CSV file   #
	read_csv()





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

