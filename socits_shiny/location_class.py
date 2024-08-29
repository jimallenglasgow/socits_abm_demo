
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

##and the function files

########################################################

##the following code defines the location class, and all functions included involving the location

class Location():

    ##the function to initialise all the values in the location

    def __init__(self, sel_location, floor_width, floor_length, no_time_steps):

        self.location_id=sel_location
        row=int(sel_location/floor_width)
        col=np.mod(sel_location,floor_width)
        self.allowed_initial_location=0
        self.possible_location=0
        self.coords=[row,col]
        self.no_agents_present=0
        self.agents_in_location=0
        self.locations_to_move_to=0
        self.is_classroom=0
        self.is_toilet=0
        self.is_staff_room=0
        self.is_canteen=0
        self.classroom_year=0
        
        ##initialise the measures in each location
        
        self.mean_location_stress=0
        self.mean_location_loneliness=0
        
        self.location_info=np.zeros([no_time_steps,2])

    ############################################################

    ##the function to check if a location is in the school map, and therefore can be moved to
    
    ##Function: Is_Possible_Location
    
    ##Inputs: school map
    
    ##Outputs: possible location
    
    ##What it does
    
    ##If a location is an allowed location, set the value of this location to 1.  If not, set it to zero.

    def Is_Possible_Location(self,school_map):

        self.possible_location=0 ##initially say that the location isn't possible

        ##now check if the coordinates are set to one in school map

        x_coord=self.coords[0]

        y_coord=self.coords[1]

        map_value=school_map[x_coord,y_coord]

        if map_value==1:

            self.possible_location=1 ##and set this as a possible location if they are


    ############################################################

    ##the function to allocate the room type
    
    def Allocate_Room_Type(self, poss_classrooms, non_classrooms):

        location_id=self.location_id

        self.is_classroom=0

        poss_location=self.possible_location
        
        ##assign the selected rooms
        
        selected_toilets=non_classrooms[0:2]
    
#        print("selected_toilets")
        
 #       print(selected_toilets)
    
        ##select a canteen
        
        selected_canteen=np.array(non_classrooms[2].astype(int))
            
        print("selected_canteen")
        
        print(selected_canteen)
            
        ##select a staff-room
        
        selected_staffroom=np.array(non_classrooms[3]).astype(int)
        
    #    print("selected_staffroom")
        
     #   print(selected_staffroom)

        ##if the location is a possible location, then what kind of location is it?

        if poss_location==1:
            
            for loc_tmp in poss_classrooms:
               
               if location_id==loc_tmp:
                   
                   self.is_classroom=1
                   
                   self.classroom_year=np.random.permutation(2)[0]+1
                   
            for loc_tmp in selected_toilets:
               
                if location_id==loc_tmp:
                   
                    self.is_toilet=1
                    
            loc_tmp=selected_canteen
               
            if location_id==loc_tmp:
               
                self.is_canteen=1
                
            loc_tmp=selected_staffroom
               
            if location_id==loc_tmp:
               
                self.is_staff_room=1
                       
            
            
            










    ############################################################

    ##the function to set whether this location is a classroom
    
    ##Function: Is_Classroom
    
    ##Inputs: grid size
    
    ##Outputs: is classroom
    
    ##What it does
    
    ##If the location is on the top or bottom of the map, then it is assigned to be a classroom

    def Is_Classroom(self,grid_size):

        poss_location=self.possible_location

        ##if the location is a possible location, then is the location on the top or bottom of the grid?

        if poss_location==1:

            coord=self.coords[1]

            if coord==0:

                self.is_classroom=1
                
            if coord==grid_size-1:

                self.is_classroom=1
                
    ############################################################
                
    ##for an oblong grid, all locations are classrooms
    
    ##Function: Is_Classroom_Oblong
    
    ##Inputs: None
    
    ##Outputs: is classroom
    
    ##What it does
    
    ##For the oblong map, all locations are classrooms
                
    def Is_Classroom_Oblong(self):

        self.is_classroom=1
        
    ############################################################

    ##this function calculates the possible locations to move to, based on the allowed locations in the location grid matrix
    
    ##Function: Calc_Locations_To_Move_To
    
    ##Inputs: location grid matrix
    
    ##Outputs: Possible locations to move to
    
    ##What it does
    
    ##This function generates all the possible locations to move to from the focal location based on the location grid matrix.

    def Calc_Locations_To_Move_To(self,location_grid_matrix):

        grid_length=len(location_grid_matrix[0,0,:])

        sel_location=self.location_id

        coords_to_move_to=np.where(location_grid_matrix[sel_location,:,:]==1) ##which locations in the grid are set to one?

        #print(coords_to_move_to)

        self.locations_to_move_to=grid_length*coords_to_move_to[0]+coords_to_move_to[1] ##set all of these as the locations to move to

        #if len(self.locations_to_move_to)>0:

        #	self.allowed_initial_location=1
        
    ############################################################

    ##this function sets a location value to 1 if it is allowed to be an initial condition (which basically means it's at the top or bottom)
    
    ##Function: Allowed_Initial_Condition_Class
    
    ##Inputs: grid size
    
    ##Outputs: allowed initial condition
    
    ##What it does
    
    ##Sets the output to 1 if the location is an allowed initial condition.

    def Allowed_Initial_Condition_Class(self,grid_size):

        loc_col=np.mod(self.location_id,grid_size)

        if loc_col==0:

            self.allowed_initial_location=1

        if loc_col==grid_size-1:

            self.allowed_initial_location=1
            
    ############################################################
            
    ##for an oblong all locations can be an initial condition
    
    ##Function: Allowed_Initial_Condition_Class_Oblong
    
    ##Inputs: None
    
    ##Outputs: allowed initial condition
    
    ##What it does
    
    ##Sets the output to 1 if the location is an allowed initial condition in the oblong map (which for is true for all locations in the oblong).
            
    def Allowed_Initial_Condition_Class_Oblong(self):

        self.allowed_initial_location=1
        
    ############################################################    
        
    ##this function calculates the number agents in the location in this time step
    
    ##Function: Calc_No_Agents_In_Location
    
    ##Inputs: all agents
    
    ##Outputs: no. of agents present
    
    ##What it does
    
    ##Calculates the number of agents present in the location.
        
    def Calc_No_Agents_In_Location(self,all_agents):

        no_agents=len(all_agents) 

        self.no_agents_present=0 ##initialise that there are no agents present

        for sel_agent in np.arange(no_agents): ##for each agent, check their location

            agent_location=all_agents[sel_agent].current_location

            if agent_location==self.location_id: ##if the agent's location matches the current location, add this to the total

                self.no_agents_present=self.no_agents_present+1
                
    ############################################################

    ##this function currently checks which agents are in the location in this time step
    
    ##Function: Which_Agents_In_Location
    
    ##Inputs: all agents
    
    ##Outputs: the agent id's present in the location
    
    ##What it does
    
    ##For a given location, this returns the agent id's of those present. 

    def Which_Agents_In_Location(self,all_agents):

        self.agents_in_location=np.zeros(self.no_agents_present) ##initialise a vector to record them

        no_agents=len(all_agents)

        agent_count=0

        for sel_agent in np.arange(no_agents): ##for all the agents...

            agent_location=all_agents[sel_agent].current_location

            if agent_location==self.location_id: ##if the agent's location matches the selected location, add the agent

                self.agents_in_location[agent_count]=sel_agent

                agent_count=agent_count+1


    ############################################################

    ##this function currently checks which agent types are in the location in this time step
    
    ##Function: Calc_No_Agent_Types_In_Location
    
    ##Inputs: all agents
    
    ##Outputs: no. agent types in the location
    
    ##What it does
    
    ##For a given location, this function tells you how many teachers and students are present.
        
    def Calc_No_Agent_Types_In_Location(self,all_agents):

        no_agents=len(all_agents)

        self.no_agent_types_present=np.zeros(3) ##initialise the empty vector for each possible agent type

        for sel_agent in np.arange(no_agents): ##for all the agents...

            agent_location=all_agents[sel_agent].current_location

            if agent_location==self.location_id: ##if the agent's location matches the selected location, add the agents type

                agent_type=all_agents[sel_agent].agent_type

                self.no_agent_types_present[int(agent_type)]=self.no_agent_types_present[int(agent_type)]+1
        
    ############################################################
        
    ##this function currently checks the status of each agent in the location in this time step
    
    ##Function: Calc_Agent_Status_In_Location
    
    ##Inputs: all agents
    
    ##Outputs: agent status
    
    ##What it does
    
    ##This function records the status of every agent in the current location.
        
    def Calc_Agent_Status_In_Location(self,all_agents):

        no_agents=len(all_agents)

        self.agent_status=np.zeros(self.no_agents_present) ##initialise an empty vector for the status of all of the agents in the location

        status_count=0 ##keep track of which agent in the location we're checking

        for sel_agent in np.arange(no_agents): ##for all the agents...

            agent_location=all_agents[sel_agent].current_location

            if agent_location==self.location_id: ##if the agent's location matches the selected location, add the agents status

                self.agent_status[status_count]=all_agents[sel_agent].status
                
                status_count=status_count+1


#################################################################

    ##function: calculate the mean stress and loneliness in the location

    def Calc_Agent_Stress_And_Loneliness_In_Location(self,all_agents):

        no_agents=len(all_agents)

        location_agent_stress_and_loneliness=np.zeros([self.no_agents_present, 2]) ##initialise an empty vector for the status of all of the agents in the location

        agent_count=0 ##keep track of which agent in the location we're checking
        
        mean_stress_loneliness=[0, 0]

        for sel_agent in np.arange(no_agents): ##for all the agents...

            agent_location=all_agents[sel_agent].current_location

            if agent_location==self.location_id: ##if the agent's location matches the selected location, add the agents status

                location_agent_stress_and_loneliness[agent_count, 0]=all_agents[sel_agent].stress
                
                location_agent_stress_and_loneliness[agent_count, 1]=all_agents[sel_agent].rq
                
                agent_count=agent_count+1
                
#        print("location_agent_stress_and_loneliness")
        
 #       print(location_agent_stress_and_loneliness)
                
        if self.no_agents_present>0:
            
            mean_stress_loneliness=np.mean(location_agent_stress_and_loneliness, axis=0)

 #       print("mean_stress_loneliness")
        
 #       print(mean_stress_loneliness)

        self.mean_location_stress=mean_stress_loneliness[0]
        self.mean_location_loneliness=mean_stress_loneliness[1]


    ##################################
    
    ##this function records all of the necessary information in this time step for this agent
 
    def Record_Location_Info(self,time):

        self.location_info[time,0]=self.mean_location_stress

        self.location_info[time,1]=self.mean_location_loneliness
        





























