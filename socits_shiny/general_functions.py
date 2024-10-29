#location: cd Desktop/socits/model/full_model/status

#to run: python3 static_network.py

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

from agent_class import Agent
from location_class import Location

########################################################

##functions

#############

##function to generate the adjacency matrix for each location

##Function: Generate_Location_Grid

##Inputs: school map; all locations

##Outputs: the location grid matrix

##What it does

##Using the school map a matrix of allowed, movable locations for each location is developed by comparing the allowed neighbours for each location.

def Generate_Location_Grid(school_map,all_locations):

    ##calculate the number of locations based on the school map

    grid_width=len(school_map[:,0])
    
    grid_length=len(school_map[0,:])

#    grid_size=grid_width*grid_length

#    print("Grid width = ",grid_width)
    
 #   print("Grid length = ",grid_length)

    no_locations=grid_width*grid_length

    location_grid_matrix=np.zeros([no_locations,grid_width,grid_length]) ##initialise the grid for all of the allowed locations to move from the specific location

    for sel_location in np.arange(no_locations): ##moving through each location

        location_coords=all_locations[sel_location].coords ##find the coordinates

        #		location_grid_matrix[sel_location,location_coords[0],location_coords[1]]=1

        ##based on the school map, set as a one places which can be moved to from the selected location
        
        ##Note: the different if loops below just include the different boundary possibilities i.e., the map is NOT a torus 

        if location_coords[0]>0:

            if school_map[location_coords[0]-1,location_coords[1]]==1:

                location_grid_matrix[sel_location,location_coords[0]-1,location_coords[1]]=1

        if location_coords[0]<(grid_width-1):

            if school_map[location_coords[0]+1,location_coords[1]]==1:

                location_grid_matrix[sel_location,location_coords[0]+1,location_coords[1]]=1

        if location_coords[1]>0:

            if school_map[location_coords[0],location_coords[1]-1]==1:

                location_grid_matrix[sel_location,location_coords[0],location_coords[1]-1]=1

        if location_coords[1]<(grid_length-1):

            if school_map[location_coords[0],location_coords[1]+1]==1:

                location_grid_matrix[sel_location,location_coords[0],location_coords[1]+1]=1

        if location_coords[0]>0 and location_coords[1]>0:

            if school_map[location_coords[0]-1,location_coords[1]-1]==1:

                location_grid_matrix[sel_location,location_coords[0]-1,location_coords[1]-1]=1

        if location_coords[0]<(grid_width-1) and location_coords[1]<(grid_length-1):

            if school_map[location_coords[0]+1,location_coords[1]+1]==1:

                location_grid_matrix[sel_location,location_coords[0]+1,location_coords[1]+1]=1

        if location_coords[1]>0 and location_coords[0]<(grid_width-1):

            if school_map[location_coords[0]+1,location_coords[1]-1]==1:

                location_grid_matrix[sel_location,location_coords[0]+1,location_coords[1]-1]=1

        if location_coords[1]<(grid_length-1) and location_coords[0]>0:

            if school_map[location_coords[0]-1,location_coords[1]+1]==1:

                location_grid_matrix[sel_location,location_coords[0]-1,location_coords[1]+1]=1

    return(location_grid_matrix)


########################################################

##function to generate a school map with a corridor of a certain size

##Function: Generate_Corridor

##Inputs: school map; grid length; grid width; corridor width

##Outputs: the updated school map

##What it does

##Adds a corridor to the school map of a certain size

def Generate_Corridor(school_map,grid_width,grid_length,corridor_width):

	left_wall=np.round(grid_width/2)-np.round(corridor_width/2) ##where should the left wall of the corridor be?

	print("left wall = ",left_wall)

	right_wall=np.round(grid_width/2)+np.floor(corridor_width/2) ##and the right?

	print("right wall = ",right_wall)

	for i in np.arange(grid_width): ##across the width

		if i<left_wall or i>right_wall: ##if the grid point falls outside the corridor...
		
	#	if i<4 or i>5:

			for j in np.arange(grid_length):
			
				if j>1 and j<(grid_length-2): ##...and is between the classrooms...
			
					school_map[i,j]=0 ##...then this is not an allowed location

	return(school_map)
    
    
    
########################################################

##function to generate a map of a floor with a corridor of a certain size

##Function: Generate_Floor_Corridor

##Inputs: school map; grid length; floor width; corridor width

##Outputs: the updated school map

##What it does

##Adds a corridor to the school map of a certain size on the ground or upper floor of the two floor school

def Generate_Floor_Corridor(school_map,grid_length,floor_width,corridor_width):

	left_wall=np.round(floor_width/2)-np.round(corridor_width/2) ##where should the left wall of the corridor be?

#	print("left wall = ",left_wall)

	right_wall=np.round(floor_width/2)+np.floor(corridor_width/2) ##and the right?

#	print("right wall = ",right_wall)

	for i in np.arange(floor_width): ##across the floor width

		if i<left_wall or i>right_wall: ##if the grid point falls outside the corridor...
		
	#	if i<4 or i>5:

			for j in np.arange(grid_length):
			
				if j>1 and j<(grid_length-2): ##...and is between the classrooms...
			
					school_map[i,j]=0 ##...then this is not an allowed location

	return(school_map)


#################################################################

##function to generate a map of a school with two identical floors

##Function: Generate_Two_Floor_School

##Inputs: floor width; stair case width; corridor width

##Outputs: school map

##What it does

##Runs the function "Generate_Floor_Corridor" twice, and sticks the results together to create a map of a two floor school with a corridor on each floor, and adds a staircase between them.

def Generate_Two_Floor_School(floor_width,stair_case_width,corridor_width):

    grid_length=2*floor_width+stair_case_width ##set the length of the grid to be the addition of both floors and the staircase

    school_map_left=np.ones([floor_width,floor_width]) ##initialise a full school map for the left hand school

    school_map_left=Generate_Floor_Corridor(school_map_left,floor_width,floor_width,corridor_width) ##and generate a map with a corridor

 #   print("school_map_left")

  #  print(school_map_left)

    school_map_right=np.ones([floor_width,floor_width]) ##initialise a full school map for the right hand school

    school_map_right=Generate_Floor_Corridor(school_map_right,floor_width,floor_width,corridor_width) ##and generate a map with a corridor

    staircase=np.zeros([stair_case_width,floor_width]) ##initialise a staircase map

    for i in np.arange(stair_case_width): ##fill in the staircase

        staircase[i,1]=1
        staircase[i,floor_width-2]=1
        
   # print("staircase")

    #print(staircase)

    school_map=np.vstack([school_map_left,staircase,school_map_right]) ##and add all these parts together to generate the school map

    return(school_map)

#################################################################

##function to generate a map of a school with two identical floors, and a closed top stairwell

##Function: Generate_Two_Floor_School_Shut_Top_Floor

##Inputs: floor width; stair case width; corridor width

##Outputs: school map

##What it does

##Runs the function "Generate_Floor_Corridor" twice, and sticks the results together to create a map of a two floor school with a corridor on each floor, and adds a staircase between them.  But in this case only the lower staircase is added so the top staircase is shut.

def Generate_Two_Floor_School_Shut_Top_Floor(floor_width,stair_case_width,corridor_width):

    grid_length=2*floor_width+stair_case_width ##set the length of the grid to be the addition of both floors and the staircase

    school_map_left=np.ones([floor_width,floor_width]) ##initialise a full school map for the left hand school

    school_map_left=Generate_Floor_Corridor(school_map_left,floor_width,floor_width,corridor_width) ##and generate a map with a corridor

 #   print("school_map_left")

  #  print(school_map_left)

    school_map_right=np.ones([floor_width,floor_width]) ##initialise a full school map for the right hand school

    school_map_right=Generate_Floor_Corridor(school_map_right,floor_width,floor_width,corridor_width) ##and generate a map with a corridor

    staircase=np.zeros([stair_case_width,floor_width]) ##initialise a staircase map

    for i in np.arange(stair_case_width): ##fill in the staircase

        staircase[i,1]=1 ##only include the bottom staircase
 #       staircase[i,grid_length-2]=1 
        
   # print("staircase")

    #print(staircase)

    school_map=np.vstack([school_map_left,staircase,school_map_right]) ##and add all these parts together to generate the school map

    return(school_map)



#################################################################

##function to generate a map of a school with two identical floors, and a closed bottom stairwell

##Function: Generate_Two_Floor_School_Shut_Bottom_Floor

##Inputs: floor width; stair case width; corridor width

##Outputs: school map

##What it does

##Runs the function "Generate_Floor_Corridor" twice, and sticks the results together to create a map of a two floor school with a corridor on each floor, and adds a staircase between them.  But in this case only the upper staircase is added so the lower staircase is shut.

def Generate_Two_Floor_School_Shut_Bottom_Floor(floor_width,stair_case_width,corridor_width):

    grid_length=2*floor_width+stair_case_width ##set the length of the grid to be the addition of both floors and the staircase

    school_map_left=np.ones([floor_width,floor_width]) ##initialise a full school map for the left hand school

    school_map_left=Generate_Floor_Corridor(school_map_left,floor_width,floor_width,corridor_width) ##and generate a map with a corridor

 #   print("school_map_left")

  #  print(school_map_left)

    school_map_right=np.ones([floor_width,floor_width]) ##initialise a full school map for the right hand school

    school_map_right=Generate_Floor_Corridor(school_map_right,floor_width,floor_width,corridor_width) ##and generate a map with a corridor

    staircase=np.zeros([stair_case_width,floor_width]) ##initialise a staircase map

    for i in np.arange(stair_case_width): ##fill in the staircase

#        staircase[i,1]=1
        staircase[i,floor_width-2]=1 ##only include the top staircase
        
   # print("staircase")

    #print(staircase)

    school_map=np.vstack([school_map_left,staircase,school_map_right]) ##and add all these parts together to generate the school map

    return(school_map)

#################################################################

##function to generate a graph linking all allowed locations based on whether a staircase is shut or not

##Function: Update_Location_Graph

##Inputs: all locations; corridors open; floor width; stair case width; corridor width

##Outputs: G (the location graph)

##What it does

##Generates a location graph of a two floor school, depending on whether the top or bottom staircase is closed.

def Update_Location_Graph(all_locations,corridors_open,floor_width,stair_case_width,corridor_width):

    ##generate the school map
    
    grid_size=2*floor_width+stair_case_width ##calculate the width of the grid
    
#    print("grid_size_1 = ",grid_size)
    
    no_locations=len(all_locations)
    
    ##generate the school map based on which staircase is open or closed

    if corridors_open==0:

        school_map=Generate_Two_Floor_School(floor_width,stair_case_width,corridor_width)

#        print("school map")
        
 #       print(school_map)

    if corridors_open==1:

        school_map=Generate_Two_Floor_School_Shut_Top_Floor(floor_width,stair_case_width,corridor_width)
        
    if corridors_open==2:

        school_map=Generate_Two_Floor_School_Shut_Bottom_Floor(floor_width,stair_case_width,corridor_width)

#    print("school_map")

 #   print(school_map)

    ##generate the location adjacency matrix

    location_grid_matrix=Generate_Location_Grid(school_map,all_locations)
        
    #print("location_grid_matrix")

    #print(location_grid_matrix)

    ##generate the points to which each individual is allowed to move, along with the classrooms

    for sel_location in np.arange(no_locations):

        all_locations[sel_location].Is_Possible_Location(school_map)
        
        all_locations[sel_location].Is_Classroom(floor_width)
        
    poss_classrooms=[]
    
    for sel_location in np.arange(no_locations): ##and then initialise them
    
        if all_locations[sel_location].is_classroom==1:
    
            poss_classrooms=np.hstack([poss_classrooms,sel_location])
    
    #################
    
    ##allocate some rooms to take certain roles e.g., toilets, staff-room
    
    ##select two classrooms to be toilets
    
    non_classrooms=[9, 49, 150, 189]#np.random.permutation(poss_classrooms)[0:4]
    
    ##and remove them from the list of possible classrooms
    
    non_class_index=np.isin(poss_classrooms, non_classrooms, invert=True)#[0]
    
#    print("non_class_index")
    
 #   print(non_class_index)
    
    poss_classrooms=poss_classrooms[non_class_index]

#    print("selected_toilets = ",selected_toilets)
    
    assigned_teacher_classrooms=np.random.permutation(poss_classrooms)
    
    print("poss_classrooms")
    
    print(poss_classrooms)
    
  #  print("Teacher classrooms")
    
   # print(assigned_teacher_classrooms)
        
    for sel_location in np.arange(no_locations): ##and then initialise them
        
        all_locations[sel_location].Allocate_Room_Type(poss_classrooms, non_classrooms)
        
        all_locations[sel_location].Calc_Locations_To_Move_To(location_grid_matrix)

        all_locations[sel_location].Allowed_Initial_Condition_Class(floor_width)
        
        

    ##using this generate the graph

    G = nx.Graph() ##initialise the graph

    G.add_nodes_from([0, grid_size*grid_size-1]) ##add the number of nodes needed

    for sel_location in np.arange(no_locations): ##for each location, look at where agents can move from this location, and then add edges to the graph for places to move to

        locations_allowed_to_move_to=all_locations[sel_location].locations_to_move_to

        for j in locations_allowed_to_move_to:

            G.add_edge(sel_location, j)
            
           
    return(G)


##############################################################################################

##Oblong map functions


#############

##function to generate the adjacency matrix for each location

##Function: Generate_Oblong_Location_Grid

##Inputs: floor width; floor length; all locations

##Outputs: the location grid matrix

##What it does

##Based on the oblong map (where every location is allowed) a matrix of allowed, movable locations for each location is developed by comparing the allowed neighbours for each location.

def Generate_Oblong_Location_Grid(floor_width,floor_length,all_locations):

    grid_size=floor_width*floor_length ##calculate the grid size

    school_map=np.ones([floor_length,floor_width]) ##initialise a full school map

#    print("Grid size = ",grid_size)

    no_locations=grid_size ##set the number of locations

 #   print("School map")
        
  #  print(school_map)
        
    location_grid_matrix=np.zeros([no_locations,floor_length,floor_width]) ##initialise the location grid matrix

#    print("location_grid_matrix")
    
 #   print(location_grid_matrix)

    for sel_location in np.arange(no_locations): ##for each location, record where the agent can move to from that location

        location_coords=all_locations[sel_location].coords
        
 #       print("Sel location = ",sel_location)
        
  #      print("Sel location coords = ",location_coords)

        #		location_grid_matrix[sel_location,location_coords[0],location_coords[1]]=1

        ##based on the school map, set as a one places which can be moved to from the selected location
        
        ##Note: the different if loops below just include the different boundary possibilities i.e., the map is NOT a torus 

        if location_coords[0]>0:

            if school_map[location_coords[0]-1,location_coords[1]]==1:

                location_grid_matrix[sel_location,location_coords[0]-1,location_coords[1]]=1

        if location_coords[0]<(floor_length-1):

            if school_map[location_coords[0]+1,location_coords[1]]==1:

                location_grid_matrix[sel_location,location_coords[0]+1,location_coords[1]]=1

        if location_coords[1]>0:

            if school_map[location_coords[0],location_coords[1]-1]==1:

                location_grid_matrix[sel_location,location_coords[0],location_coords[1]-1]=1

        if location_coords[1]<(floor_width-1):

            if school_map[location_coords[0],location_coords[1]+1]==1:

                location_grid_matrix[sel_location,location_coords[0],location_coords[1]+1]=1

        if location_coords[0]>0 and location_coords[1]>0:

            if school_map[location_coords[0]-1,location_coords[1]-1]==1:

                location_grid_matrix[sel_location,location_coords[0]-1,location_coords[1]-1]=1

        if location_coords[0]<(floor_length-1) and location_coords[1]<(floor_width-1):

            if school_map[location_coords[0]+1,location_coords[1]+1]==1:

                location_grid_matrix[sel_location,location_coords[0]+1,location_coords[1]+1]=1

        if location_coords[1]>0 and location_coords[0]<(floor_length-1):

            if school_map[location_coords[0]+1,location_coords[1]-1]==1:

                location_grid_matrix[sel_location,location_coords[0]+1,location_coords[1]-1]=1

        if location_coords[1]<(floor_width-1) and location_coords[0]>0:

            if school_map[location_coords[0]-1,location_coords[1]+1]==1:

                location_grid_matrix[sel_location,location_coords[0]-1,location_coords[1]+1]=1

    return(location_grid_matrix)
    
#########################################################

##function to generate the graph for an oblong map

##Function: Generate_Oblong_Graph

##Inputs: floor width; floor length; location grid matrix; all_locations

##Outputs: G (the location graph)

##What it does

##Based on the location grid matrix, this function generates the graph on which to base the movements in an oblong map.

def Generate_Oblong_Graph(floor_width,floor_length,location_grid_matrix,all_locations):

    school_map=np.ones([floor_length,floor_width]) ##initialise the school map

    grid_size=floor_width*floor_length

    no_locations=grid_size

    ##generate the points to which each individual is allowed to move, along with the classrooms

    for sel_location in np.arange(no_locations):

        all_locations[sel_location].Is_Possible_Location(school_map)
        
        all_locations[sel_location].Is_Classroom_Oblong()
        
    poss_classrooms=[]
    
    for sel_location in np.arange(no_locations): ##and then initialise them
    
        if all_locations[sel_location].is_classroom==1:
    
            poss_classrooms=np.hstack([poss_classrooms,sel_location])
    
    #################
    
    ##allocate some rooms to take certain roles e.g., toilets, staff-room
    
    ##select two classrooms to be toilets
    
    non_classrooms=np.random.permutation(poss_classrooms)[0:4]
    
    ##and remove them from the list of possible classrooms
    
    non_class_index=np.isin(poss_classrooms, non_classrooms, invert=True)#[0]
    
#    print("non_class_index")
    
 #   print(non_class_index)
    
    poss_classrooms=poss_classrooms[non_class_index]

#    print("selected_toilets = ",selected_toilets)
    
    assigned_teacher_classrooms=np.random.permutation(poss_classrooms)
    
    print("poss_classrooms")
    
    print(poss_classrooms)
    
  #  print("Teacher classrooms")
    
   # print(assigned_teacher_classrooms)
        
    for sel_location in np.arange(no_locations): ##and then initialise them
        
        all_locations[sel_location].Allocate_Room_Type(poss_classrooms, non_classrooms)

        all_locations[sel_location].Calc_Locations_To_Move_To(location_grid_matrix)

        all_locations[sel_location].Allowed_Initial_Condition_Class_Oblong()

    ##using this generate a the graph

    G = nx.Graph() ##initialise the graph

    G.add_nodes_from([0, grid_size-1]) ##add the correct number of nodes

    for sel_location in np.arange(no_locations): ##for each location, look at where agents can move from this location, and then add edges to the graph for places to move to

        locations_allowed_to_move_to=all_locations[sel_location].locations_to_move_to

        for j in locations_allowed_to_move_to:

            G.add_edge(sel_location, j)
            
           
    return(G)













#######################################################################################################

def Assign_Groups(all_agents, no_groups):

    no_agents=len(all_agents)
 
    if no_groups>no_agents:
        
        no_groups=no_agents

#    print("no_agents=",no_agents)

    ##code to assign movement groups

    assigned_groups=np.ones(no_agents)*-1

    #print("assigned_groups")

    #print(assigned_groups)

    max_per_group=int(no_agents/no_groups)#+1

    remaining_group_slots=np.ones(no_groups)*max_per_group

    #print("remaining_group_slots")

    #print(remaining_group_slots)

    ########

    ##first group members

    all_poss_group_members=np.random.permutation(no_agents)

    initial_group_members=all_poss_group_members[0:no_groups]

    assigned_groups[initial_group_members]=np.arange(no_groups)

    #print("assigned_groups")

    #print(assigned_groups)

    remaining_group_slots=remaining_group_slots-1

    later_group_members=all_poss_group_members[no_groups:no_agents]

    #next_ind=later_group_members[0]#np.random.permutation(np.where(assigned_groups==-1)[0])[0]

    for next_ind in later_group_members:

     #   print("next_ind = ",next_ind)

        next_ind_rq=all_agents[next_ind].all_rq

     #   print("next_ind_rq")

    #    print(next_ind_rq)

        not_allowed_next_groups=np.where(remaining_group_slots<1)[0]

     #   print("not_allowed_next_groups")

      #  print(not_allowed_next_groups)

        group_rq=np.zeros(no_groups)

        for sel_group in np.arange(no_groups):

            sel_group_members=np.where(assigned_groups==sel_group)[0]
            
    #        print("sel_group_members")
            
     #       print(sel_group_members)
            
            sel_group_rq=next_ind_rq[sel_group_members]

      #      print("sel_group_rq")
            
       #     print(sel_group_rq)
            
            group_rq[sel_group]=np.mean(sel_group_rq)
            
        group_rq[not_allowed_next_groups]=-1000+np.random.random(len(not_allowed_next_groups))
        
       # print("group_rq")
        
    #    print(group_rq)

        next_group=np.argmax(group_rq)

     #   print("next_group = ",next_group)

        assigned_groups[next_ind]=next_group

    #    print("assigned_groups")

     #   print(assigned_groups)

        remaining_group_slots[next_group]=remaining_group_slots[next_group]-1

    #    print("remaining_group_slots")

     #   print(remaining_group_slots)

    for sel_agent in np.arange(no_agents):
        
        all_agents[sel_agent].movement_group=int(assigned_groups[sel_agent])
     
    return(assigned_groups)

##############################################################################################

def Set_New_Group_Goals(no_groups, canteen_prob, current_lunch_time, all_agents, no_teachers, no_in_each_age):
    
    all_group_goals=np.zeros(no_groups)
    
    for sel_group in np.arange(no_groups):
        
        if sel_group<no_groups/2:
        
            poss_goal_locations=all_agents[(no_teachers+1)].all_classrooms ##the goals are the classrooms
        
        else:
            
            older_agent=no_in_each_age[0]+no_teachers+1
            
            poss_goal_locations=all_agents[(older_agent+1)].all_classrooms ##the goals are the classrooms
        
        sel_goal=int(np.random.permutation(poss_goal_locations)[0]) ##select one at random

        if current_lunch_time==1:
            
            r=np.random.random()
        
            if r<canteen_prob:
                
                sel_goal=all_agents[0].all_canteens[0]
                

        all_group_goals[sel_group]=sel_goal

#    print("all_group_goals")
    
 #   print(all_group_goals)
    
    no_agents=len(all_agents)
    
    for sel_agent in np.arange(no_agents):
        
        all_agents[sel_agent].group_goals=all_group_goals



























