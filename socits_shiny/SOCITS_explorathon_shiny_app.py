##shiny run --reload --launch-browser GitHub/socits_abm_demo/socits_shiny/SOCITS_explorathon_shiny_app.py

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

from general_functions import Generate_Location_Grid
from general_functions import Generate_Corridor
from general_functions import Generate_Floor_Corridor
from general_functions import Generate_Two_Floor_School
from general_functions import Generate_Two_Floor_School_Shut_Top_Floor
from general_functions import Generate_Two_Floor_School_Shut_Bottom_Floor

from general_functions import Update_Location_Graph

#from plot_output_functions import Plot_Output_Shiny

from run_single_model_function import Run_The_Model_Once

from shiny.express import ui, input, render
from shiny import reactive


with ui.sidebar():

#    ui.input_slider("no_students", "No. students", 1, 200, 100)
    
#    ui.input_slider("no_teachers", "No. teachers", 1, 50, 10)
    
    ui.input_slider("stress_decay", "Stress decay", 0, 1, 0.1)
    
    ui.input_checkbox("inc_science_perspective_input", "Include sci persp.", True)
    
    ui.input_checkbox("inc_youth_perspective_input", "Include youth persp.", True)
    
#    ui.input_checkbox("inc_walking_perspective_input", "Include walk persp.", False)
    
 #   ui.input_checkbox("inc_yais_perspective_input", "Include YAIs persp.", False)
    
  #  ui.input_checkbox("inc_teacher_perspective_input", "Include teacher persp.", False)
    
    ui.input_radio_buttons("output_type", "Output", choices=["Stress", "Loneliness"])
    
    ui.input_radio_buttons("plot_type", "Plot type", choices=["Heat", "Map", "Time"])
    
    ui.input_slider("sel_time_step", "Selected time step", 1, 100, 50)
    
    ui.input_radio_buttons("run_new", "Run new simulation", choices=["Yes", "No"])
    
@render.plot(alt="A histogram")

def Socits_Model():

    ##general variables

    ##run the model code?

    ##1=run the model before plotting

    ##0=load data from the last run
    
    use_emp_networks=0

    ##map size

    ##map type options....

    ##"grid" = standard grid
    ##"school" = the school map

    map_type="school"

    floor_width=10

    floor_length=10

    stair_case_width=5

    corridor_width=1

    ##time steps

    no_time_steps=100

    ##no. of different agent types

    no_students=100#input.no_students()

    no_teachers=10#input.no_teachers()

    no_agents=no_students+no_teachers

    no_bullies=0
    
    no_groups=30

    ##parameters common to all perspectives

    stress_decay=input.stress_decay() ##how quickly stress decays

    prob_teacher_moving=1 ##the probability of a teacher moving around the school
    
    move_with_friends=1 ##should individuals try to move with friends?  1==Yes, 0==No

    moving_times=[25, 50, 75]

    lunch_times=[0, 1, 0]

    mean_time_stress=1 ##the mean of the normal distribution for the time stress

    mean_room_stress=1
    
    toilet_prob=0.01 ##probability of any student going to the toilet at any one time

    canteen_prob=0.4 ##probability of any student eating lunch in the canteen

    prob_follow_group=0.5

    ##parameters to include certain perspective (1=include)

    inc_science_perspective=0
    
    inc_walking_perspective=0

    inc_yais_perspective=0

    inc_teacher_perspective=0

    if input.inc_science_perspective_input()==1:

        inc_science_perspective=1
    
    else:
    
        inc_science_perspective=0

    if input.inc_youth_perspective_input()==1:

        inc_walking_perspective=1
        
        inc_yais_perspective=1
        
        inc_teacher_perspective=1
    
    else:
    
        inc_walking_perspective=0
        
        inc_yais_perspective=0
        
        inc_teacher_perspective=0


#    if input.inc_walking_perspective_input()==1:

 #       inc_walking_perspective=1
    
  #  else:
    
#        inc_walking_perspective=0
        
 #   if input.inc_yais_perspective_input()==1:

  #      inc_yais_perspective=1
    
   # else:
    
    #    inc_yais_perspective=0
        
#    if input.inc_teacher_perspective_input()==1:

 #       inc_teacher_perspective=1
    
  #  else:
    
   #     inc_teacher_perspective=0


    ##science model

    ##triggers

    status_threshold=1 ##theshold for the status difference before a negative interaction

    stress_bully_scale=1

    ##consequences

    increase_in_stress_due_to_neg_int=1 ##change in stress due to a negative interaction

    decrease_in_stress_due_to_pos_int=0.2 ##change in stress due to a positive interaction

    rq_decrease_through_bullying=1 ##change in relationship quality due to a negative interaction

    rq_increase_through_support=1.5 ##change in relationship quality due to a positive interaction

    status_increase=1 ##increase in status if an individual interacts negatively

    status_decrease=1 ##decrease in status if an individual receives a negative interaction

    ##yai's model

    ##triggers

    crowded_threshold=3 ##number of others in a location before stress increases

    ##consequences

    crowded_stress=2 ##the amount by which stress increases due to crowdedness

    journey_stress=1 ##stress increase if a journey is disrupted

    ##walking interviews model

    ##No additional parameters required

    ##teachers model

    ##triggers

    no_classes=4 ##no. of different possible classes

    class_prob_dist=[0.25,0.25,0.25,0.25] ##how these classes are distributed in the population

    ##consequences

    stress_through_class_interaction=0.4 ##stress increase due to negative interaction between students and teachers

    ########################################################

    ##Part C: run the model

    ##the following line calls the function "Run_The_Model_Once" in the file "run_single_model_function.py" to run the model, and returns all of the agents information in each time step
        
    if input.run_new()=="Yes":

        plot_type="0"
        
        all_technical_inputs=[plot_type, map_type, floor_width, floor_length, stair_case_width, corridor_width, no_time_steps, no_students, no_teachers, no_bullies,inc_science_perspective, inc_walking_perspective, inc_yais_perspective, inc_teacher_perspective, no_classes, moving_times, move_with_friends, lunch_times, no_groups, class_prob_dist, use_emp_networks]

        all_inputs_set_through_data=[toilet_prob, canteen_prob]

        all_calibrated_inputs=[stress_decay, status_threshold, increase_in_stress_due_to_neg_int, decrease_in_stress_due_to_pos_int, rq_decrease_through_bullying,rq_increase_through_support, crowded_threshold, crowded_stress, journey_stress, stress_bully_scale, stress_through_class_interaction, prob_teacher_moving, status_increase, status_decrease, mean_time_stress, mean_room_stress, prob_follow_group]

        all_outputs=Run_The_Model_Once(all_calibrated_inputs, all_inputs_set_through_data, all_technical_inputs)

        all_agents=all_outputs[0]
    
        all_locations=all_outputs[1]

        ###########

        ##save the model output

        ##first generate an array which contains each agent's info at each timestep

        sel_agent=0

        model_output=np.hstack([np.zeros([no_time_steps,1]),all_agents[sel_agent].agent_info])

        for sel_agent in np.arange(1,no_agents):

            sel_agent_model_output=np.hstack([np.ones([no_time_steps,1])*sel_agent,all_agents[sel_agent].agent_info])

            model_output=np.vstack([model_output,sel_agent_model_output])

        print("Model output")

        print(model_output)

        ##and write them to a file

#        file_name=Path(__file__).parent/"model_output_shiny.csv"

        file_name="GitHub/socits_abm/model_output_shiny.csv"

        no_values=len(model_output[:,0])

        with open(file_name, 'w', encoding='UTF8',newline='') as f:

            writer = csv.writer(f)
            
            for sol_value in np.arange(no_values): 

                writer.writerow(model_output[sol_value,:])

    if input.run_new()=="No":
        
        model_output_file=open('GitHub/socits_abm/model_output_shiny.csv' %locals())
        model_output_tmp=csv.reader(model_output_file)
        model_output=list(model_output_tmp)
        model_output=np.array(model_output)

        model_output=model_output.astype(float)

        print("Model output")

        print(model_output)
        
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
        
        all_agents=[]

        for sel_agent in np.arange(no_agents):

            all_agents.append(Agent(sel_agent, no_time_steps, age_category))

        for sel_agent in np.arange(no_agents):
    
            sel_agent_info_loc=np.where(model_output[:,0]==sel_agent)[0]
            
            sel_agent_info_all_cols=model_output[sel_agent_info_loc,:]
            
            sel_agent_info=sel_agent_info_all_cols[:,[1,2,3,4,5,6]]
            
            all_agents[sel_agent].agent_info=sel_agent_info
            
    
    
    ########
    
    ##plot_type.....

    ##0 = animation

    ##1 = single time step

    ##3 = selected output

    plot_type="3"

    sel_time_step=input.sel_time_step()

    ##selected output.....

    ##3 = stress

    ##4 = loneliness

    if input.output_type()=="Stress":

        selected_output=3
        
    elif input.output_type()=="Loneliness":

        selected_output=4

    selected_output_name="Stress"
    #selected_output_name="Relationship quality"

    save_animation="No"

#    Plot_Output_Shiny(all_agents, floor_width, floor_length, stair_case_width, corridor_width, plot_type, selected_output, selected_output_name, map_type, save_animation, sel_time_step)

    plot_type=input.plot_type()

    if plot_type=="Time":

        no_agents=len(all_agents) ##find the number of agents

        no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

        all_agent_output=np.zeros([no_time_steps,no_agents]) ##initialise agent output for all agents

        all_agent_types=np.zeros(no_agents) ##and record their types

        for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

            all_agent_output[:,sel_agent]=all_agents[sel_agent].agent_info[:,selected_output]
            
            all_agent_types[sel_agent]=all_agents[sel_agent].agent_info[0,2] ##and the types
            

        student_agents=np.where(all_agent_types==0)[0] ##find all the student agents

    #        print("student_agents")

    #       print(student_agents)

        #all_agent_output=all_agent_output[:,student_agents]

        min_all_agent_output=np.min(np.mean(all_agent_output[:,student_agents],axis=0)) ##find the minimum mean output

        max_all_agent_output=np.max(np.mean(all_agent_output[:,student_agents],axis=0)) ##and the maximum

    #      print("min_agent_stress")

    #     print(min_all_agent_output)

    #    print("max_agent_stress")

     #   print(max_all_agent_output)

        sel_min_agent=np.where([np.mean(all_agent_output[:,student_agents],axis=0)==min_all_agent_output])[1][0] ##find which agent has the minimum

        sel_max_agent=np.where([np.mean(all_agent_output[:,student_agents],axis=0)==max_all_agent_output])[1][0] ##and the maximum

      #  print("min agent = ",sel_min_agent)

       # print("max agent = ",sel_max_agent)

        perm_agents=np.random.permutation(student_agents) ##randomly permute the agents so we can select some random ones

        no_students=len(student_agents)
        
        if no_students<4: ##if there is a small number of students, plot them all
        
            sel_agents=perm_agents

        else: ##if not, select two at random and the smallest and largest output agents

            sel_agents=np.hstack([perm_agents[0:2],sel_min_agent,sel_max_agent])

    #        print("sel_agents")

    #       print(sel_agents)

        #print("all_agent_output")

        #print(all_agent_output)
        
        ##plot the output across time
        
        fig, ax = plt.subplots()

        for plot_agent in sel_agents: ##plot each agent output

            ax.plot(np.arange(no_time_steps), all_agent_output[:,plot_agent])

        ax.set_xlabel("Time")

        ax.set_ylabel(selected_output_name)
    
    ##display it and save it
    
    if plot_type=="Map":
    
        no_agents=len(all_agents) ##find the number of agents

        no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

        all_agent_output=np.zeros([no_time_steps,no_agents]) ##initialise agent output for all agents

        all_agent_types=np.zeros(no_agents) ##and record their types

        for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

            all_agent_output[:,sel_agent]=all_agents[sel_agent].agent_info[:,selected_output]
            
            all_agent_types[sel_agent]=all_agents[sel_agent].agent_info[0,2] ##and the types
            
        print("All agent output")
        
        print(all_agent_output)
            
        max_output=np.max(all_agent_output) ##find the maximum output across all agents, to be scaled for later

        min_output=np.min(all_agent_output)

        print("max_output")

        print(max_output)
        
        ##and then scale the agent output so it is between 0 and 1
        
        all_agent_output_scaled=(all_agent_output-min_output)/(max_output-min_output)

        print("All agent output scaled")
        
        print(all_agent_output_scaled)

        ##initialise the output plot

        fig, ax = plt.subplots()

        ##generate a scatter for all agents locations

        current_agent_locations=np.zeros([no_agents,4]) ##initialise the agent information array for this time step

        poss_colours=np.array(['b','r']) ##initialise the possible colours
        
        for sel_agent in np.arange(no_agents):

            current_agent_locations[sel_agent,[0,1]]=all_agents[sel_agent].agent_info[sel_time_step,[0,1]]

        current_agent_locations[:,2]=all_agent_output_scaled[sel_time_step,:]
        
        current_agent_locations[:,3]=all_agent_types

        print("Current agent locations")

        print(current_agent_locations)
        
        z=current_agent_locations[:,2]#.astype(int) ##and the value of the output of interest

        ##create a matrix to generate a colour of each agent

        colour_matrix=np.zeros([no_agents,4]) 

        reds=z

        blues=1-z

        greens=current_agent_locations[:,3]/2

        alpha=np.ones(no_agents)

        colour_matrix[:,0]=reds

        colour_matrix[:,1]=greens

        colour_matrix[:,2]=blues

        colour_matrix[:,3]=alpha

        print("colours")

        print(colour_matrix)


#        scat=ax.scatter(all_agents[0].agent_info[0,0], all_agents[0].agent_info[0,1], c=[[current_agent_locations[0,2],0,0,1]], s=30) ##plot the initial scatter plot

        scat=ax.scatter(current_agent_locations[:,0], current_agent_locations[:,1], c=colour_matrix, s=30) ##plot the initial scatter plot

        #########################################
        
        if map_type=="grid":
        
            ##grid walls

            for sel_grid in np.arange(floor_width+1):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[-0.5,floor_length+1.5],color='k',linewidth=2)
                
            for sel_grid in np.arange(floor_length+2):

                ax.plot([-0.5,floor_width+1.5],[sel_grid-0.5,sel_grid-0.5],color='k',linewidth=2)
        
        ##########

            ax.set(xlim=[-0.5, floor_length-0.5], ylim=[-0.5, floor_width-0.5])

        if map_type=="school":        
        
            ##plot the school map

            ##plot the ground floor

            initial_floor_wall=0

            left_wall=initial_floor_wall+np.round(floor_width/2)-np.round(corridor_width/2)

            right_wall=initial_floor_wall+np.round(floor_width/2)+np.floor(corridor_width/2)+1

            ##lower classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[0.5,0.5],color='k',linewidth=2)

            ##upper classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[floor_length-1.5,floor_length-1.5],color='k',linewidth=2)

            ##classroom walls

            for sel_grid in np.arange(initial_floor_wall,initial_floor_wall+floor_width+1):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[-0.5,0.5],color='k',linewidth=2)
                
                ax.plot([sel_grid-0.5,sel_grid-0.5],[floor_length-1.5,floor_length-0.5],color='k',linewidth=2)
            
            ##left top corridor wall
                
            ax.plot([initial_floor_wall-0.5,left_wall-1.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##left bottom corridor wall
            
            ax.plot([initial_floor_wall-0.5,left_wall-1.5],[2.5,2.5],color='k',linewidth=2)

            ##right top corridor wall

            ax.plot([right_wall,initial_floor_wall+floor_width-0.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##right bottom corridor wall

            ax.plot([right_wall,initial_floor_wall+floor_width-0.5],[2.5,2.5],color='k',linewidth=2)

            ##corridor right wall

            ax.plot([right_wall,right_wall],[2.5,floor_length-3.5],color='k',linewidth=2)
            
            ##corridor left wall

            ax.plot([left_wall-1.5,left_wall-1.5],[2.5,floor_length-3.5],color='k',linewidth=2)
            
            ##########################
            
            ##plot the upper floor

            initial_floor_wall=floor_width+stair_case_width

            left_wall=initial_floor_wall+np.round(floor_width/2)-np.round(corridor_width/2)

            right_wall=initial_floor_wall+np.round(floor_width/2)+np.floor(corridor_width/2)+1

            ##lower classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[0.5,0.5],color='k',linewidth=2)

            ##upper classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[floor_length-1.5,floor_length-1.5],color='k',linewidth=2)

            ##classroom walls

            for sel_grid in np.arange(initial_floor_wall,initial_floor_wall+floor_width+1):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[-0.5,0.5],color='k',linewidth=2)
                
                ax.plot([sel_grid-0.5,sel_grid-0.5],[floor_length-1.5,floor_length-0.5],color='k',linewidth=2)
            
            ##left top corridor wall
                
            ax.plot([initial_floor_wall-0.5,left_wall-1.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##left bottom corridor wall
            
            ax.plot([initial_floor_wall-0.5,left_wall-1.5],[2.5,2.5],color='k',linewidth=2)

            ##right top corridor wall

            ax.plot([right_wall,initial_floor_wall+floor_width-0.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##right bottom corridor wall

            ax.plot([right_wall,initial_floor_wall+floor_width-0.5],[2.5,2.5],color='k',linewidth=2)

            ##corridor right wall

            ax.plot([right_wall,right_wall],[2.5,floor_length-3.5],color='k',linewidth=2)
            
            ##corridor left wall

            ax.plot([left_wall-1.5,left_wall-1.5],[2.5,floor_length-3.5],color='k',linewidth=2)
            
            ##########
            
            ##plot the staircase
            
            staircase_beginning=floor_width
            
            staircase_end=staircase_beginning+stair_case_width
            
            ##top staircase upper wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[floor_length-1.5,floor_length-1.5],color='k',linewidth=2)
            
            ##top staircase lower wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##bottom staircase upper wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[2.5,2.5],color='k',linewidth=2)
            
            ##bottom staircase lower wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[0.5,0.5],color='k',linewidth=2)
            
            ##and add some stairs

            for sel_grid in np.arange(staircase_beginning+0.5,staircase_end,0.25):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[0.5,2.5],color='k',linewidth=2)
                
                ax.plot([sel_grid-0.5,sel_grid-0.5],[floor_length-3.5,floor_length-1.5],color='k',linewidth=2)
            
            ##########

            ax.set(xlim=[-0.5, floor_width*2+stair_case_width-0.5], ylim=[-0.5, floor_length-0.5])
        
        ##and animate the output

        plt.xticks([])
        plt.yticks([])
        
        
    if plot_type=="Heat":
    
        no_agents=len(all_agents) ##find the number of agents

        no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

        all_agent_output=np.zeros([no_time_steps,no_agents]) ##initialise agent output for all agents

        all_agent_types=np.zeros(no_agents) ##and record their types

        for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

            all_agent_output[:,sel_agent]=all_agents[sel_agent].agent_info[:,selected_output]
            
            all_agent_types[sel_agent]=all_agents[sel_agent].agent_info[0,2] ##and the types
            
        print("All agent output")
        
        print(all_agent_output)
            
        max_output=np.max(all_agent_output) ##find the maximum output across all agents, to be scaled for later

        min_output=np.min(all_agent_output)

        print("max_output")

        print(max_output)
        
        ##and then scale the agent output so it is between 0 and 1
        
        all_agent_output_scaled=(all_agent_output-min_output)/(max_output-min_output)

        print("All agent output scaled")
        
        print(all_agent_output_scaled)

        ##initialise the output plot

        fig, ax = plt.subplots()

        ##generate a scatter for all agents locations

        current_agent_locations=np.zeros([no_agents,4]) ##initialise the agent information array for this time step

        poss_colours=np.array(['b','r']) ##initialise the possible colours
        
        for sel_time_step in np.arange(no_time_steps):
        
            for sel_agent in np.arange(no_agents):

                current_agent_locations[sel_agent,[0,1]]=all_agents[sel_agent].agent_info[sel_time_step,[0,1]]

            current_agent_locations[:,2]=all_agent_output_scaled[sel_time_step,:]
            
            current_agent_locations[:,3]=all_agent_types

            print("Current agent locations")

            print(current_agent_locations)
            
            z=current_agent_locations[:,2]#.astype(int) ##and the value of the output of interest

            ##create a matrix to generate a colour of each agent

            colour_matrix=np.zeros([no_agents,4]) 

            reds=z

            blues=1-z

            greens=current_agent_locations[:,3]/2

            alpha=np.ones(no_agents)

            colour_matrix[:,0]=reds

            colour_matrix[:,1]=greens

            colour_matrix[:,2]=blues

            colour_matrix[:,3]=alpha

            print("colours")

            print(colour_matrix)


    #        scat=ax.scatter(all_agents[0].agent_info[0,0], all_agents[0].agent_info[0,1], c=[[current_agent_locations[0,2],0,0,1]], s=30) ##plot the initial scatter plot

            scat=ax.scatter(current_agent_locations[:,0], current_agent_locations[:,1], c=colour_matrix, s=10) ##plot the initial scatter plot

        #########################################
        
        if map_type=="grid":
        
            ##grid walls

            for sel_grid in np.arange(floor_width+1):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[-0.5,floor_length+1.5],color='k',linewidth=2)
                
            for sel_grid in np.arange(floor_length+2):

                ax.plot([-0.5,floor_width+1.5],[sel_grid-0.5,sel_grid-0.5],color='k',linewidth=2)
        
        ##########

            ax.set(xlim=[-0.5, floor_length-0.5], ylim=[-0.5, floor_width-0.5])

        if map_type=="school":        
        
            ##plot the school map

            ##plot the ground floor

            initial_floor_wall=0

            left_wall=initial_floor_wall+np.round(floor_width/2)-np.round(corridor_width/2)

            right_wall=initial_floor_wall+np.round(floor_width/2)+np.floor(corridor_width/2)+1

            ##lower classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[0.5,0.5],color='k',linewidth=2)

            ##upper classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[floor_length-1.5,floor_length-1.5],color='k',linewidth=2)

            ##classroom walls

            for sel_grid in np.arange(initial_floor_wall,initial_floor_wall+floor_width+1):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[-0.5,0.5],color='k',linewidth=2)
                
                ax.plot([sel_grid-0.5,sel_grid-0.5],[floor_length-1.5,floor_length-0.5],color='k',linewidth=2)
            
            ##left top corridor wall
                
            ax.plot([initial_floor_wall-0.5,left_wall-1.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##left bottom corridor wall
            
            ax.plot([initial_floor_wall-0.5,left_wall-1.5],[2.5,2.5],color='k',linewidth=2)

            ##right top corridor wall

            ax.plot([right_wall,initial_floor_wall+floor_width-0.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##right bottom corridor wall

            ax.plot([right_wall,initial_floor_wall+floor_width-0.5],[2.5,2.5],color='k',linewidth=2)

            ##corridor right wall

            ax.plot([right_wall,right_wall],[2.5,floor_length-3.5],color='k',linewidth=2)
            
            ##corridor left wall

            ax.plot([left_wall-1.5,left_wall-1.5],[2.5,floor_length-3.5],color='k',linewidth=2)
            
            ##########################
            
            ##plot the upper floor

            initial_floor_wall=floor_width+stair_case_width

            left_wall=initial_floor_wall+np.round(floor_width/2)-np.round(corridor_width/2)

            right_wall=initial_floor_wall+np.round(floor_width/2)+np.floor(corridor_width/2)+1

            ##lower classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[0.5,0.5],color='k',linewidth=2)

            ##upper classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[floor_length-1.5,floor_length-1.5],color='k',linewidth=2)

            ##classroom walls

            for sel_grid in np.arange(initial_floor_wall,initial_floor_wall+floor_width+1):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[-0.5,0.5],color='k',linewidth=2)
                
                ax.plot([sel_grid-0.5,sel_grid-0.5],[floor_length-1.5,floor_length-0.5],color='k',linewidth=2)
            
            ##left top corridor wall
                
            ax.plot([initial_floor_wall-0.5,left_wall-1.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##left bottom corridor wall
            
            ax.plot([initial_floor_wall-0.5,left_wall-1.5],[2.5,2.5],color='k',linewidth=2)

            ##right top corridor wall

            ax.plot([right_wall,initial_floor_wall+floor_width-0.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##right bottom corridor wall

            ax.plot([right_wall,initial_floor_wall+floor_width-0.5],[2.5,2.5],color='k',linewidth=2)

            ##corridor right wall

            ax.plot([right_wall,right_wall],[2.5,floor_length-3.5],color='k',linewidth=2)
            
            ##corridor left wall

            ax.plot([left_wall-1.5,left_wall-1.5],[2.5,floor_length-3.5],color='k',linewidth=2)
            
            ##########
            
            ##plot the staircase
            
            staircase_beginning=floor_width
            
            staircase_end=staircase_beginning+stair_case_width
            
            ##top staircase upper wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[floor_length-1.5,floor_length-1.5],color='k',linewidth=2)
            
            ##top staircase lower wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##bottom staircase upper wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[2.5,2.5],color='k',linewidth=2)
            
            ##bottom staircase lower wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[0.5,0.5],color='k',linewidth=2)
            
            ##and add some stairs

            for sel_grid in np.arange(staircase_beginning+0.5,staircase_end,0.25):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[0.5,2.5],color='k',linewidth=2)
                
                ax.plot([sel_grid-0.5,sel_grid-0.5],[floor_length-3.5,floor_length-1.5],color='k',linewidth=2)
            
            ##########

            ax.set(xlim=[-0.5, floor_width*2+stair_case_width-0.5], ylim=[-0.5, floor_length-0.5])
        
        ##and animate the output

        plt.xticks([])
        plt.yticks([])
