########################################################

##Part A: load in the libraries and functions for running the code

##libraries

import random
from random import randint
import numpy as np
import csv
import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import networkx as nx

from functools import partial

import scipy as sp

import pandas as pd

##and the function files

from agent_class import Agent
from location_class import Location

from general_functions import Generate_Oblong_Location_Grid
from general_functions import Generate_Oblong_Graph
from general_functions import Generate_Corridor
from general_functions import Generate_Floor_Corridor
from general_functions import Generate_Two_Floor_School
from general_functions import Generate_Two_Floor_School_Shut_Top_Floor
from general_functions import Generate_Two_Floor_School_Shut_Bottom_Floor

from general_functions import Update_Location_Graph

from general_functions import Assign_Groups

from general_functions import Set_New_Group_Goals

#from plot_output_functions import Plot_Output

#######################################################################################################################################################

##function to run a single run of the model

def Run_The_Model_Once(all_calibrated_inputs, all_inputs_set_through_data, all_technical_inputs):
    
    ##set parameters
    
    plot_type=all_technical_inputs[0]
    map_type=all_technical_inputs[1]
    floor_width=all_technical_inputs[2]
    floor_length=all_technical_inputs[3]
    stair_case_width=all_technical_inputs[4]
    corridor_width=all_technical_inputs[5]
    no_time_steps=all_technical_inputs[6]
    no_students=all_technical_inputs[7]
    no_teachers=all_technical_inputs[8]
    no_bullies=all_technical_inputs[9]
    inc_science_perspective=all_technical_inputs[10]
    inc_walking_perspective=all_technical_inputs[11]
    inc_yais_perspective=all_technical_inputs[12]
    inc_teacher_perspective=all_technical_inputs[13]
    no_classes=all_technical_inputs[14]
    
    moving_times=all_technical_inputs[15]
    move_with_friends=all_technical_inputs[16]
    lunch_times=all_technical_inputs[17]
    
    no_groups=all_technical_inputs[18]
    
    class_prob_dist=all_technical_inputs[19]
    
    use_emp_networks=all_technical_inputs[20]
    
    no_initially_stressed_students=all_technical_inputs[21]
    
    ##parameters set by the data
    
    toilet_prob=all_inputs_set_through_data[0]
    canteen_prob=all_inputs_set_through_data[1]
    
    ##parameters to be calibrated
    
    stress_decay=all_calibrated_inputs[0]
    status_threshold=all_calibrated_inputs[1]
    increase_in_stress_due_to_neg_int=all_calibrated_inputs[2]
    decrease_in_stress_due_to_pos_int=all_calibrated_inputs[3]
    rq_decrease_through_bullying=all_calibrated_inputs[4]
    rq_increase_through_support=all_calibrated_inputs[5]
    
    crowded_threshold=all_calibrated_inputs[6]
    crowded_stress=all_calibrated_inputs[7]
    journey_stress=all_calibrated_inputs[8]
    stress_bully_scale=all_calibrated_inputs[9]
    
    stress_through_class_interaction=all_calibrated_inputs[10]
    prob_teacher_moving=all_calibrated_inputs[11]
    status_increase=all_calibrated_inputs[12]
    status_decrease=all_calibrated_inputs[13]
    
    mean_time_stress=all_calibrated_inputs[14]
    mean_room_stress=all_calibrated_inputs[15]
    
    prob_follow_group=all_calibrated_inputs[16]
    
    #############
    
    ##load in the empirical networks
    
    emp_networks=0

    print("emp_networks")

    print(emp_networks)
    
    ##Part xxx: initialise the locations

    grid_size=0
    
    if map_type=="grid":

        grid_size=floor_width*floor_length ##size of the grid 
        
        no_locations=floor_width*floor_length ##one location for each grid point

    if map_type=="school":

        grid_width=2*floor_width+stair_case_width ##width of the grid
        
        grid_length=floor_width ##length of the grid

        no_locations=grid_width*grid_length ##one location for each grid point

    all_locations=[] ##initialise the locations
    
#        if input.update_geography()=="1":
    
    
    
    for sel_location in np.arange(no_locations): ##and then initialise them
    
        all_locations.append(Location(sel_location, floor_width, floor_length, no_time_steps))
 
    ##commented code to check that it worked
 
#    sel_location=np.random.randint(no_locations)
    
#    print("Random location ID")
    
#    print(all_locations[sel_location].location_id)
    
#    print("Random location coordinate")
    
#    print(all_locations[sel_location].coords)
    
    ######################
    
    ##Part xxx: generate the grid to look like a rectangle or the map of the school
    
        ##set up the information for each agent
    
    if map_type=="grid":
    
        ##update the allowed locations and the graph connecting them based on the school map
        
        location_grid_matrix=Generate_Oblong_Location_Grid(floor_width,floor_length,all_locations)
        
        ##display it
        
#        print("location_grid_matrix")
        
 #       print(location_grid_matrix)
        
        ##generate the graph of connected locations
        
        G=Generate_Oblong_Graph(floor_width,floor_length,location_grid_matrix,all_locations)
        
#        G_top_closed=Generate_Oblong_Graph(floor_width,floor_length,location_grid_matrix,all_locations)
        
 #       G_bottom_closed=Generate_Oblong_Graph(floor_width,floor_length,location_grid_matrix,all_locations)


    if map_type=="school":

        corridors_open=0
        
        G=Update_Location_Graph(all_locations,corridors_open,floor_width,stair_case_width,corridor_width)
                
        corridors_open=1
                
#        G_top_closed=Update_Location_Graph(all_locations,corridors_open,floor_width,stair_case_width,corridor_width)
        
        corridors_open=2
                
 #       G_bottom_closed=Update_Location_Graph(all_locations,corridors_open,floor_width,stair_case_width,corridor_width)

                    
    #########################
    
    ##Part xxx: initialise the agents
                    
    no_agents=no_students+no_teachers
    
    no_locations=len(all_locations)
        
    ##assign agents to be bullies and teachers

    ##type 0 = student
    ##type 1 = bully
    ##type 2 = teacher

    initial_agent_type_ordered=np.zeros(no_agents)

    initial_agent_type_ordered[0:no_teachers]=2

    initial_agent_type_ordered[no_teachers:(no_teachers+no_bullies)]=1

    #print("initial_agent_type_ordered")

    #print(initial_agent_type_ordered)

    initial_agent_types=initial_agent_type_ordered#np.random.permutation(initial_agent_type_ordered)

    #print("initial_agent_type")

    #print(initial_agent_types)#
    
    #########
    
    ##assign ages to each agent
    
    age_category=np.zeros(no_agents)
    
    if use_emp_networks==1:
    
        no_in_each_age=[171, 205]
        
    else:
        
        no_in_each_age_tmp=int((no_agents-no_teachers)/2)
        
        no_in_each_age=np.zeros(2).astype(int)
        
        no_in_each_age[0]=int(no_in_each_age_tmp)
        
        no_in_each_age[1]=int(no_in_each_age_tmp)
        
    age_category[no_teachers:(no_in_each_age[0]+no_teachers)]=1
    
    age_category[(no_in_each_age[0]+no_teachers+1):no_agents]=2
    
    
        
        
    
    #########
    
    all_agents=[] ##initialise the agents

    ##and then generate them

    for sel_agent in np.arange(no_agents):

        all_agents.append(Agent(sel_agent, no_time_steps, age_category))

    for sel_agent in np.arange(no_agents):

        all_agents[sel_agent].Record_All_Location_Types(all_locations) ##which are the classrooms

        all_agents[sel_agent].Initialise_Agent_Type(initial_agent_types) ##what type are they
        
        all_agents[sel_agent].Initialise_Agent_Class(no_classes,class_prob_dist) ##what class are they?

    ##permute all the classrooms so that we can assign them to teachers

    poss_classrooms=all_agents[0].all_classrooms

#    print("poss_classrooms")
    
 #   print(poss_classrooms)

    assigned_teacher_classrooms=np.random.permutation(poss_classrooms)

  #  print("assigned_teacher_classrooms")
    
   # print(assigned_teacher_classrooms)

    for sel_agent in np.arange(no_agents):

        all_agents[sel_agent].Initialise_RQ(all_agents, emp_networks, initial_agent_types, no_teachers, use_emp_networks) ##initialise the social network
        
    S2_agent_ids=np.where(age_category==1)[0]
    
    S4_agent_ids=np.where(age_category==2)[0]
    
    print("S4_agents")
    
    print(S4_agent_ids)
        
    S2_agents=[]
        
    for i in S2_agent_ids:
        
            S2_agents.append(all_agents[i])
            
    S4_agents=[]
        
    for i in S4_agent_ids:
        
            S4_agents.append(all_agents[i])
        
    assigned_groups=Assign_Groups(S2_agents, int(no_groups/2))
    
    assigned_groups=Assign_Groups(S4_agents, int(no_groups/2))

    print("assigned_groups")

    print(assigned_groups)
    
    for i in S4_agent_ids:
        
            all_agents[i].movement_group=all_agents[i].movement_group+int(no_groups/2)

#    print("Example groups")
    
 #   print(all_agents[50].movement_group)
    
  #  print(all_agents[250].movement_group)

    Set_New_Group_Goals(no_groups, canteen_prob, 0, all_agents, no_teachers, no_in_each_age)

    for sel_agent in np.arange(no_agents):

        all_agents[sel_agent].Initialise_Location(all_locations,assigned_teacher_classrooms, prob_follow_group) ##where do they start?

        all_agents[sel_agent].Initialise_Agent_Location_Time_Stress(all_locations,no_time_steps, mean_time_stress, mean_room_stress, inc_walking_perspective, inc_yais_perspective) ##how stressed are they in their initial location

        all_agents[sel_agent].Record_Agent_Info(0) ##save this initial information

        
 #   print("All classrooms")

  #  print(all_agents[0].all_classrooms)
        
    ##also initialise the information in the locations
        
    for sel_location in np.arange(no_locations):

            all_locations[sel_location].Calc_No_Agents_In_Location(all_agents) ##how many agents are in the location?
            
            all_locations[sel_location].Calc_No_Agent_Types_In_Location(all_agents) ##how many agents of each type are there?
            
            all_locations[sel_location].Calc_Agent_Status_In_Location(all_agents) ##what is the status of all the agents in that location?
            
    ##finally, initialise the next goal location of the agent
            
    for sel_agent in np.arange(no_agents):

        ################

        ##movement functions

        all_agents[sel_agent].Set_New_Goal_Class(prob_teacher_moving, canteen_prob, 0, prob_follow_group, assigned_teacher_classrooms) ##where is the new goal?

        all_agents[sel_agent].Ideal_Goal_Length(all_locations,G) ##how long should it take?

    #    print("goal length = ",all_agents[sel_agent].ideal_goal_length)

        ################

        ##stress functions

#        all_agents[sel_agent].Decide_If_Stressed_Due_To_Time(all_locations,status_threshold,0)

 #       all_agents[sel_agent].Decide_If_Stressed_Due_To_Location(all_locations,status_threshold,0)


        ################

        ##record the info

        all_agents[sel_agent].Record_Agent_Info(0)


    ###########################################
    
    ##Part xxx: run the model for a set amount of time

    ##and now move the agent around

    G_current=G ##set the current map of the school
    
    sel_moving_time=0
    
    first_moving_time=moving_times[sel_moving_time]
    
    no_moving_times=len(moving_times)
    
    for time in np.arange(1,no_time_steps): ##run through all the time steps

#        if np.mod(int((time/no_time_steps)*100),10)==0:

        first_moving_time=moving_times[0]
        
        if time==first_moving_time:
            
                all_students=np.random.permutation(np.arange(no_teachers,no_agents))
                
                sel_stressed_students=all_students[0:(int(no_initially_stressed_students)+1)]
                
#                print("sel_stressed_students")
                
 #               print(sel_stressed_students)
 
                for sel_agent in sel_stressed_students:
                    
                    all_agents[sel_agent].stress=3

        current_moving_time=moving_times[sel_moving_time]
        
        current_lunch_time=lunch_times[sel_moving_time] ##is it currently lunch time?
        
        print("%age done = ",np.round(time/no_time_steps*100),"%", end="\r")

        ################

        ##movement functions

        ##and run through the actions of the agent
        
        ##decide if an agent should go to the toilet
            
        for sel_agent in np.arange(no_agents):

            agent_type=all_agents[sel_agent].agent_type ##agent type
            
            if agent_type==1:
            
                r=np.random.random()
                
                if r<toilet_prob: ##...and by chance they don't move
                    
                    all_agents[sel_agent].Set_New_Goal_Toilet()

        ##initially don't move the agent for the first ten time steps, but after that move them every 20
        
        if time>first_moving_time:
        
            ##if the time is larger than the first moving time, move the agent

            for sel_agent in np.arange(no_agents):

                all_agents[sel_agent].Move_Agent(all_locations,G_current)
            
        ##move the agent every set time steps, and reset the goal and the ideal goal length

        if time==current_moving_time:
        
            if sel_moving_time<no_moving_times-1:
        
                sel_moving_time=sel_moving_time+1
            
            current_moving_time=moving_times[sel_moving_time]
            
            assigned_teacher_classrooms=np.random.permutation(poss_classrooms)

            Set_New_Group_Goals(no_groups, canteen_prob, current_lunch_time, all_agents, no_teachers, no_in_each_age)

            for sel_agent in np.arange(no_agents):

                all_agents[sel_agent].Set_New_Goal_Class(prob_teacher_moving, canteen_prob, current_lunch_time, prob_follow_group, assigned_teacher_classrooms)

                all_agents[sel_agent].Ideal_Goal_Length(all_locations,G)



        ##update the location information after these movements

        for sel_location in np.arange(no_locations):

            all_locations[sel_location].Calc_No_Agents_In_Location(all_agents)

            all_locations[sel_location].Calc_No_Agent_Types_In_Location(all_agents)

            all_locations[sel_location].Calc_Agent_Status_In_Location(all_agents)

            all_locations[sel_location].Which_Agents_In_Location(all_agents)
            
            all_locations[sel_location].Calc_Agent_Stress_And_Loneliness_In_Location(all_agents)
            
            all_locations[sel_location].Record_Location_Info(time)

        ################
        
        ##decide which interactions occur

        ##negative interactions

        for sel_agent in np.arange(no_agents):

            all_agents[sel_agent].Decide_Negative_Interaction(all_agents,all_locations,status_threshold,stress_bully_scale)

            all_agents[sel_agent].Decide_Negative_Class_Interaction(all_agents,all_locations)

        ##################

        ##positive interactions (possibly in response to bullying)

        for sel_agent in np.arange(no_agents):

            all_agents[sel_agent].Decide_Positive_Interaction(all_agents,all_locations)

        ################

        ##update stress, rq and status based on which perspectives are included
        
        if inc_science_perspective==1:

            for sel_agent in np.arange(no_agents):

                all_agents[sel_agent].Update_Stress_Due_To_Interactions(all_locations,all_agents,increase_in_stress_due_to_neg_int,decrease_in_stress_due_to_pos_int)

                all_agents[sel_agent].Update_Status(status_increase,status_decrease)

                all_agents[sel_agent].Update_Loneliness_Calc()

                all_agents[sel_agent].Update_All_RQ_Based_On_Interaction(rq_decrease_through_bullying,rq_increase_through_support)

        if inc_walking_perspective==1 or inc_yais_perspective==1:

            for sel_agent in np.arange(no_agents):
                
                all_agents[sel_agent].Decide_If_Stressed_Due_To_Time(all_locations,status_threshold,time)

                all_agents[sel_agent].Decide_If_Stressed_Due_To_Location(all_locations,status_threshold,time)

                all_agents[sel_agent].Decide_If_Stressed_Due_To_Crowdedness(all_locations,crowded_threshold,crowded_stress)
                
        if inc_yais_perspective==1:

            for sel_agent in np.arange(no_agents):

                #print("Journey time = ",all_agents[sel_agent].journey_time,", Ideal journey time = ",all_agents[sel_agent].ideal_goal_length)
                
                all_agents[sel_agent].Decide_If_Stressed_Due_To_Delay(journey_stress)

            
        if inc_teacher_perspective==1:
        
            for sel_agent in np.arange(no_agents):

                all_agents[sel_agent].Update_Stress_Due_To_Class_Interactions(stress_through_class_interaction,decrease_in_stress_due_to_pos_int)
            
        for sel_agent in np.arange(no_agents):
            
            all_agents[sel_agent].Decay_Stress(stress_decay, time)
            
            all_agents[sel_agent].Calc_Network_Degree()
            
            all_agents[sel_agent].Calc_Situational_Loneliness(all_locations)

        ################

        ##record the info

        for sel_agent in np.arange(no_agents):

            all_agents[sel_agent].Record_Agent_Info(time)
                
        all_outputs=[all_agents, all_locations]
                
    return(all_outputs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
