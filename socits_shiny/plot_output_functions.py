##libraries

import random
from random import randint
import numpy as np
import csv
import time
import os
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import networkx as nx

from functools import partial

import scipy as sp

##and the function files

#from agent_class import Agent
#from location_class import Location

#from general_functions import Generate_Location_Grid
#from general_functions import Generate_Corridor

#from run_single_model_function import Run_A_Single_Run_Of_The_Model


#######################################################################################################################################################

##function to run a single run of the model

##Function:

##Inputs:

##Outputs:

##What it does

##

def Plot_Output(all_agents, all_locations, model_inputs, plot_type, selected_output, selected_output_name, map_type, save_animation, sel_time_step):
    
    inc_science_perspective=int(model_inputs[0])
    inc_walking_perspective=int(model_inputs[1])
    inc_yais_perspective=int(model_inputs[2])
    inc_teacher_perspective=int(model_inputs[3])
    stress_decay=model_inputs[4]
    status_threshold=model_inputs[5]
    increase_in_stress_due_to_neg_int=model_inputs[6]
    decrease_in_stress_due_to_pos_int=model_inputs[7]
    rq_decrease_through_bullying=model_inputs[8]
    rq_increase_through_support=model_inputs[9]
    crowded_threshold=int(model_inputs[10])
    crowded_stress=model_inputs[11]
    journey_stress=model_inputs[12]
    stress_bully_scale=model_inputs[13]
    stress_through_class_interaction=model_inputs[14]
    prob_teacher_moving=model_inputs[15]
    status_increase=model_inputs[16]
    status_decrease=model_inputs[17]
    mean_time_stress=model_inputs[18]
    mean_room_stress=model_inputs[19]
    prob_follow_group=model_inputs[20]
    teacher_standards_mean=model_inputs[21]
    teacher_standards_sd=model_inputs[22]
    student_standards_mean=model_inputs[23]
    student_standards_sd=model_inputs[24]
    standards_interaction_scale=model_inputs[25]
    initial_status_mean=model_inputs[26]
    initial_status_sd=model_inputs[27]
    reduction_due_to_teacher_presence=model_inputs[28]
    rq_mean=model_inputs[29]
    rq_sd=model_inputs[30]
    initial_stress_scale=model_inputs[31]
    stair_case_width=int(model_inputs[32])
    corridor_width=int(model_inputs[33])
    classroom_size=int(model_inputs[34])
    no_classrooms_per_corridor=int(model_inputs[35])
    floor_length=int(model_inputs[36])
    toilets_loc_1=int(model_inputs[37])
    toilets_loc_2=int(model_inputs[38])
    canteen_loc=int(model_inputs[39])
    staffroom_loc=int(model_inputs[40])
    no_time_steps=int(model_inputs[41])
    no_students=int(model_inputs[42])
    no_bullies=int(model_inputs[43])
    no_initially_stressed_students=int(model_inputs[44])
    no_boys_in_each_age=int(model_inputs[45])
    no_girls_in_each_age=int(model_inputs[46])
    class_length=int(model_inputs[47])
    toilet_prob=model_inputs[48]
    canteen_prob=model_inputs[49]
    
    non_classrooms=[toilets_loc_1, toilets_loc_2, canteen_loc, staffroom_loc]
    
    floor_width=no_classrooms_per_corridor*classroom_size+(no_classrooms_per_corridor-1)

#    all_agents=Run_A_Single_Run_Of_The_Model(corridor_width=corridor_width,grid_size=grid_size)

    if plot_type=="3": ##plot the evolution of the selected output

        no_agents=len(all_agents) ##find the number of agents

        no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

        all_agent_output=np.zeros([no_time_steps,no_agents]) ##initialise agent output for all agents

        all_agent_types=np.zeros(no_agents) ##and record their types

        for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

            all_agent_output[:,sel_agent]=all_agents[sel_agent].agent_info[:,selected_output]
            
            all_agent_types[sel_agent]=all_agents[sel_agent].agent_info[0,2] ##and the types
            

        student_agents=np.where(all_agent_types==0)[0] ##find all the student agents

        print("student_agents")

        print(student_agents)

        #all_agent_output=all_agent_output[:,student_agents]

        min_all_agent_output=np.min(np.mean(all_agent_output[:,student_agents],axis=0)) ##find the minimum mean output

        max_all_agent_output=np.max(np.mean(all_agent_output[:,student_agents],axis=0)) ##and the maximum

        print("min_agent_stress")

        print(min_all_agent_output)

        print("max_agent_stress")

        print(max_all_agent_output)

        sel_min_agent=np.where([np.mean(all_agent_output[:,student_agents],axis=0)==min_all_agent_output])[1][0] ##find which agent has the minimum

        sel_max_agent=np.where([np.mean(all_agent_output[:,student_agents],axis=0)==max_all_agent_output])[1][0] ##and the maximum

        print("min agent = ",sel_min_agent)

        print("max agent = ",sel_max_agent)

        perm_agents=np.random.permutation(student_agents) ##randomly permute the agents so we can select some random ones

        no_students=len(student_agents)
        
        if no_students<4: ##if there is a small number of students, plot them all
        
            sel_agents=perm_agents

        else: ##if not, select two at random and the smallest and largest output agents

            sel_agents=np.hstack([perm_agents[0:2],sel_min_agent,sel_max_agent])

        print("sel_agents")

        print(sel_agents)

        #print("all_agent_output")

        #print(all_agent_output)
        
        ##plot the output across time
        
        fig, ax = plt.subplots()

        for plot_agent in sel_agents: ##plot each agent output

            ax.plot(np.arange(no_time_steps), all_agent_output[:,plot_agent])

        ax.set_xlabel("Time")

        ax.set_ylabel(selected_output_name)
        
        ##display it and save it

        plt.show()
        
        fig.savefig(f"agent_output_{selected_output_name}.png")
        
        plt.close()
    

    ###############################################################
    
    if plot_type=="4": ##plot the evolution of the selected output in the location

        no_locations=len(all_locations) ##find the number of agents

        no_time_steps=len(all_locations[0].location_info[:,0]) ##and the number of time steps

        no_locations=len(all_locations) ##and the number of time steps

        all_location_output=np.zeros([no_time_steps,no_locations]) ##initialise agent output for all agents

        for sel_location in np.arange(no_locations): ##for each agent, assign the specific output to the output array

            all_location_output[:,sel_location]=all_locations[sel_location].location_info[:,selected_output-3]
            
        min_all_location_output=np.min(np.mean(all_location_output,axis=0)) ##find the minimum mean output

        max_all_location_output=np.max(np.mean(all_location_output,axis=0)) ##and the maximum

        print("min_location_stress")

        print(min_all_location_output)

        print("max_location_stress")

        print(max_all_location_output)

        sel_min_location=np.where([np.mean(all_location_output,axis=0)==min_all_location_output])[1][0] ##find which agent has the minimum

        sel_max_location=np.where([np.mean(all_location_output,axis=0)==max_all_location_output])[1][0] ##and the maximum

        print("min location = ",sel_min_location)

        print("max location = ",sel_max_location)

        perm_locations=np.random.permutation(np.arange(no_locations)) ##randomly permute the agents so we can select some random ones

        if no_locations<4: ##if there is a small number of students, plot them all
        
            sel_locations=perm_locations

        else: ##if not, select two at random and the smallest and largest output agents

            sel_locations=np.hstack([perm_locations[0:2],sel_min_location, sel_max_location])

        print("sel_locations")

        print(sel_locations)

        #print("all_agent_output")

        #print(all_agent_output)
        
        ##plot the output across time
        
        fig, ax = plt.subplots()

        for plot_location in sel_locations: ##plot each agent output

            ax.plot(np.arange(no_time_steps), all_location_output[:,plot_location])

        ax.set_xlabel("Time")

        ax.set_ylabel(selected_output_name)
        
        ##display it and save it

        plt.show()
        
        fig.savefig(f"location_output_{selected_output_name}.png")
        
        plt.close()
    

    ###############################################################
    
    ##plot the distribution
    
    if plot_type=="5": ##plot the distribution of the selected output

        no_agents=len(all_agents) ##find the number of agents

        no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

        all_agent_output=np.zeros([no_time_steps,no_agents]) ##initialise agent output for all agents

        all_agent_types=np.zeros(no_agents) ##and record their types

        for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

            all_agent_output[:,sel_agent]=all_agents[sel_agent].agent_info[:,selected_output]
            
            all_agent_types[sel_agent]=all_agents[sel_agent].agent_info[0,2] ##and the types
            

        student_agents=np.where(all_agent_types==0)[0] ##find all the student agents

        print("student_agents")

        print(student_agents)

        #all_agent_output=all_agent_output[:,student_agents]

        final_time_output=all_agent_output[no_time_steps-1, student_agents]
        
        fig, ax = plt.subplots()

        ax.hist(final_time_output)

        ax.set_xlabel(selected_output_name)

        ax.set_ylabel("Frequency")
        
        ##display it and save it

        plt.show()
        
        fig.savefig(f"agent_output_dist_{selected_output_name}.png")
        
        plt.close()
        

    if plot_type=="0":
    
        ###############
    
    	##the function to animate the agents
    
        def update(frame):

            current_agent_locations=np.zeros([no_agents,4]) ##initialise the agent information array for this time step

            poss_colours=np.array(['b','r']) ##initialise the possible colours
            
            for sel_agent in np.arange(no_agents):

                current_agent_locations[sel_agent,[0,1]]=all_agents[sel_agent].agent_info[frame,[0,1]]

            current_agent_locations[:,2]=all_agent_output_scaled[frame,:]
            
            current_agent_locations[:,3]=all_agent_types

            x = current_agent_locations[:,0] ##find the x-coordinates
            y = current_agent_locations[:,1] ##and the y

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

            #	print("colours")

            #	print(colour_matrix)

            # update the scatter plot:
            
            data = np.stack([x, y]).T

            scat.set_offsets(data)
            scat.set_color(colour_matrix)

            return (scat)
            
        ###############
    
        ##now run the animation plotting
        
        print("Animation")
            
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

#        print("max_output")

 #       print(max_output)
        
        ##and then scale the agent output so it is between 0 and 1
        
        all_agent_output_scaled=(all_agent_output-min_output)/(max_output-min_output)

  #      print("All agent output scaled")
        
   #     print(all_agent_output_scaled)

        ##initialise the output plot

        fig, ax = plt.subplots()

        ##generate a scatter for all agents locations

        current_agent_locations=np.zeros([no_agents,4]) ##initialise the agent information array for this time step

        poss_colours=np.array(['b','r']) ##initialise the possible colours
        
        for sel_agent in np.arange(no_agents):

            current_agent_locations[sel_agent,[0,1]]=all_agents[sel_agent].agent_info[0,[0,1]]

        current_agent_locations[:,2]=all_agent_output_scaled[0,:]
        
        current_agent_locations[:,3]=all_agent_types

#        print("Current agent locations")

 #       print(current_agent_locations)
        
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

 #       print("colours")

  #      print(colour_matrix)

#        scat=ax.scatter(all_agents[0].agent_info[0,0], all_agents[0].agent_info[0,1], c=[[current_agent_locations[0,2],0,0,1]], s=30) ##plot the initial scatter plot

        scat=ax.scatter(current_agent_locations[:,0], current_agent_locations[:,1], c=colour_matrix, s=30) ##plot the initial scatter plot

        #########################################
        
        ##plot the school map

#        floor_width=int(np.round((grid_size-stair_case_width)/2))

        if map_type=="grid":
        
            ##grid walls

            for sel_grid in np.arange(floor_width+1):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[-0.5,floor_length+1.5],color='k',linewidth=2)
                
            for sel_grid in np.arange(floor_length+2):

                ax.plot([-0.5,floor_width+1.5],[sel_grid-0.5,sel_grid-0.5],color='k',linewidth=2)
        
            ##########

            ax.set(xlim=[-0.5, floor_length-0.5], ylim=[-0.5, floor_width-0.5])

        if map_type=="school":

            ##########################
            
            ##plot the ground floor

            initial_floor_wall=0

            left_wall=initial_floor_wall+np.round(floor_width/2)-np.round(corridor_width/2)

            right_wall=initial_floor_wall+np.round(floor_width/2)+np.floor(corridor_width/2)+1

            ##lower classroom wall
                
            ax.plot([initial_floor_wall-1.5,initial_floor_wall+floor_width-0.5],[classroom_size-0.5,classroom_size-0.5],color='k',linewidth=2)

            ##upper classroom wall
                
            ax.plot([initial_floor_wall-1.5,initial_floor_wall+floor_width-0.5],[floor_length-classroom_size-0.5,floor_length-classroom_size-0.5],color='k',linewidth=2)

            ##classroom walls

            for sel_classroom in np.arange(no_classrooms_per_corridor+1):

                ax.plot([sel_classroom*(classroom_size+1)-1.5,sel_classroom*(classroom_size+1)-1.5],[-0.5,classroom_size-0.5],color='k',linewidth=2)
                
                ax.plot([sel_classroom*(classroom_size+1)-1.5,sel_classroom*(classroom_size+1)-1.5],[floor_length-classroom_size-0.5,floor_length-0.5],color='k',linewidth=2)
            
            ##left top corridor wall
                
            ax.plot([initial_floor_wall-1,left_wall-1.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##left bottom corridor wall
            
            ax.plot([initial_floor_wall-1,left_wall-1.5],[2.5,2.5],color='k',linewidth=2)

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
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[classroom_size-0.5,classroom_size-0.5],color='k',linewidth=2)

            ##upper classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[floor_length-classroom_size-0.5,floor_length-classroom_size-0.5],color='k',linewidth=2)

            ##classroom walls
            
            for sel_classroom in np.arange(no_classrooms_per_corridor+1):

                ax.plot([floor_width+stair_case_width+sel_classroom*(classroom_size+1)-1.5, floor_width+stair_case_width+sel_classroom*(classroom_size+1)-1.5],[-0.5,classroom_size-0.5],color='k',linewidth=2)
                
                ax.plot([floor_width+stair_case_width+sel_classroom*(classroom_size+1)-1.5, floor_width+stair_case_width+sel_classroom*(classroom_size+1)-1.5],[floor_length-classroom_size-0.5,floor_length-0.5],color='k',linewidth=2)

            
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
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[floor_length-classroom_size-0.5,floor_length-classroom_size-0.5],color='k',linewidth=2)
            
            ##top staircase lower wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##bottom staircase upper wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[2.5,2.5],color='k',linewidth=2)
            
            ##bottom staircase lower wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[classroom_size-0.5,classroom_size-0.5],color='k',linewidth=2)
            
            ##and add some stairs

            for sel_grid in np.arange(staircase_beginning+0.5,staircase_end,0.25):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[classroom_size-0.5,2.5],color='k',linewidth=2)
                
                ax.plot([sel_grid-0.5,sel_grid-0.5],[floor_length-3.5,floor_length-classroom_size-0.5],color='k',linewidth=2)
            
            ##########

#            ax.set(xlim=[-1.5, floor_width*2+stair_case_width+0.5], ylim=[-1.5, floor_length+0.5])
        
        ##and animate the output

        plt.xticks([])
        plt.yticks([])

        ani = animation.FuncAnimation(fig=fig, func=partial(update), frames=no_time_steps, interval=(10/no_time_steps)*1000)
        plt.show()
        
        ##and finally save the output as a gif
        
        if save_animation=="Yes":
        
            f = r"socits_model.gif" 
            writergif = animation.PillowWriter(fps=5) 
            ani.save(f, writer=writergif)
        
#        ani.save(filename="socits_model.mp4", writer="ffmpeg")

    if plot_type=="1":
    
        ###############
    
        no_agents=len(all_agents) ##find the number of agents

        no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

        all_agent_output=np.zeros([no_time_steps,no_agents]) ##initialise agent output for all agents

        all_agent_types=np.zeros(no_agents) ##and record their types

        for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

            all_agent_output[:,sel_agent]=all_agents[sel_agent].agent_info[:,selected_output]
            
            all_agent_types[sel_agent]=all_agents[sel_agent].agent_info[0,2] ##and the types
            
#        print("All agent output")
        
 #       print(all_agent_output)
            
        max_output=np.max(all_agent_output) ##find the maximum output across all agents, to be scaled for later

        min_output=np.min(all_agent_output)

  #      print("max_output")

   #     print(max_output)
        
        ##and then scale the agent output so it is between 0 and 1
        
        all_agent_output_scaled=(all_agent_output-min_output)/(max_output-min_output)

    #    print("All agent output scaled")
        
     #   print(all_agent_output_scaled)

        ##initialise the output plot

        fig, ax = plt.subplots()

        ##generate a scatter for all agents locations

        current_agent_locations=np.zeros([no_agents,4]) ##initialise the agent information array for this time step

        poss_colours=np.array(['b','r']) ##initialise the possible colours
        
        for sel_agent in np.arange(no_agents):

            current_agent_locations[sel_agent,[0,1]]=all_agents[sel_agent].agent_info[sel_time_step,[0,1]]

        current_agent_locations[:,2]=all_agent_output_scaled[sel_time_step,:]
        
        current_agent_locations[:,3]=all_agent_types

      #  print("Current agent locations")

       # print(current_agent_locations)
        
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

            ##########################
            
            ##plot the ground floor

            initial_floor_wall=0

            left_wall=initial_floor_wall+np.round(floor_width/2)-np.round(corridor_width/2)

            right_wall=initial_floor_wall+np.round(floor_width/2)+np.floor(corridor_width/2)+1

            ##lower classroom wall
                
            ax.plot([initial_floor_wall-1.5,initial_floor_wall+floor_width-0.5],[classroom_size-0.5,classroom_size-0.5],color='k',linewidth=2)

            ##upper classroom wall
                
            ax.plot([initial_floor_wall-1.5,initial_floor_wall+floor_width-0.5],[floor_length-classroom_size-0.5,floor_length-classroom_size-0.5],color='k',linewidth=2)

            ##classroom walls

            for sel_classroom in np.arange(no_classrooms_per_corridor+1):

                ax.plot([sel_classroom*(classroom_size+1)-1.5,sel_classroom*(classroom_size+1)-1.5],[-0.5,classroom_size-0.5],color='k',linewidth=2)
                
                ax.plot([sel_classroom*(classroom_size+1)-1.5,sel_classroom*(classroom_size+1)-1.5],[floor_length-classroom_size-0.5,floor_length-0.5],color='k',linewidth=2)
            
            ##left top corridor wall
                
            ax.plot([initial_floor_wall-1,left_wall-1.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##left bottom corridor wall
            
            ax.plot([initial_floor_wall-1,left_wall-1.5],[2.5,2.5],color='k',linewidth=2)

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
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[classroom_size-0.5,classroom_size-0.5],color='k',linewidth=2)

            ##upper classroom wall
                
            ax.plot([initial_floor_wall-0.5,initial_floor_wall+floor_width-0.5],[floor_length-classroom_size-0.5,floor_length-classroom_size-0.5],color='k',linewidth=2)

            ##classroom walls
            
            for sel_classroom in np.arange(no_classrooms_per_corridor+1):

                ax.plot([floor_width+stair_case_width+sel_classroom*(classroom_size+1)-1.5, floor_width+stair_case_width+sel_classroom*(classroom_size+1)-1.5],[-0.5,classroom_size-0.5],color='k',linewidth=2)
                
                ax.plot([floor_width+stair_case_width+sel_classroom*(classroom_size+1)-1.5, floor_width+stair_case_width+sel_classroom*(classroom_size+1)-1.5],[floor_length-classroom_size-0.5,floor_length-0.5],color='k',linewidth=2)

            
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
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[floor_length-classroom_size-0.5,floor_length-classroom_size-0.5],color='k',linewidth=2)
            
            ##top staircase lower wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[floor_length-3.5,floor_length-3.5],color='k',linewidth=2)
            
            ##bottom staircase upper wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[2.5,2.5],color='k',linewidth=2)
            
            ##bottom staircase lower wall
            
            ax.plot([staircase_beginning-0.5,staircase_end-0.5],[classroom_size-0.5,classroom_size-0.5],color='k',linewidth=2)
            
            ##and add some stairs

            for sel_grid in np.arange(staircase_beginning+0.5,staircase_end,0.25):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[classroom_size-0.5,2.5],color='k',linewidth=2)
                
                ax.plot([sel_grid-0.5,sel_grid-0.5],[floor_length-3.5,floor_length-classroom_size-0.5],color='k',linewidth=2)
            
            ##########

#            ax.set(xlim=[-1.5, floor_width*2+stair_case_width+0.5], ylim=[-1.5, floor_length+0.5])
        
        ##and animate the output

        plt.xticks([])
        plt.yticks([])

        plt.show()
        
        fig.savefig(f"single_time_step_{selected_output_name}.png")
        
    if plot_type=="2":
    
        ###############
    
        ##now plot all stress across all time
            
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

#        print("max_output")

 #       print(max_output)
        
        ##and then scale the agent output so it is between 0 and 1
        
        all_agent_output_scaled=(all_agent_output-min_output)/(max_output-min_output)

  #      print("All agent output scaled")
        
   #     print(all_agent_output_scaled)

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

    #        print("Current agent locations")

     #       print(current_agent_locations)
            
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

     #       print("colours")

      #      print(colour_matrix)

    #        scat=ax.scatter(all_agents[0].agent_info[0,0], all_agents[0].agent_info[0,1], c=[[current_agent_locations[0,2],0,0,1]], s=30) ##plot the initial scatter plot

            scat=ax.scatter(current_agent_locations[:,0], current_agent_locations[:,1], c=colour_matrix, s=3) ##plot the initial scatter plot

        #########################################
        
        ##plot the school map

#        floor_width=int(np.round((grid_size-stair_case_width)/2))

        if map_type=="grid":
        
            ##grid walls

            for sel_grid in np.arange(floor_width+1):

                ax.plot([sel_grid-0.5,sel_grid-0.5],[-0.5,floor_length+1.5],color='k',linewidth=2)
                
            for sel_grid in np.arange(floor_length+2):

                ax.plot([-0.5,floor_width+1.5],[sel_grid-0.5,sel_grid-0.5],color='k',linewidth=2)
        
            ##########

            ax.set(xlim=[-0.5, floor_length-0.5], ylim=[-0.5, floor_width-0.5])

        if map_type=="school":

            ##########################
            
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

        plt.show()
        
        fig.savefig(f"heat_map.png")
        
        plt.close()
        
    if plot_type=="6": ##plot the SOCITS squares from the model
        
        no_agents=len(all_agents) ##find the number of agents
        
        no_locations=len(all_locations)

        no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

        all_SAM2_output=np.zeros([no_agents, 4]) ##initialise agent output for all agents
        
        all_SAM2_count=np.ones([no_agents, 4]) ##initialise how many time steps this data is for
        
        poss_classrooms=all_agents[0].all_classrooms
        
        poss_toilets=all_agents[0].all_toilets
        
        poss_canteen=all_agents[0].all_canteens
        
        print("poss_classrooms")
        
        print(poss_classrooms)
        
        print("poss_toilets")
        
        print(poss_toilets)
        
        print("poss_canteen")
        
        print(poss_canteen)
        
        all_non_corridors=np.hstack([poss_classrooms, poss_toilets, poss_canteen])
        
        print("all_non_corridors")
        
        print(all_non_corridors)

        for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

            sel_agent_info=all_agents[sel_agent].agent_info
            
            for time_step in np.arange(no_time_steps):
                
                agent_loc=int(sel_agent_info[time_step, 8])-1
                
                assigned_loc=0
                
                is_non_corridor=np.isin(agent_loc, all_non_corridors)
                
                if is_non_corridor==1:
                    
                    is_class=np.isin(agent_loc, poss_classrooms)
                    
                    is_canteen=np.isin(agent_loc, poss_canteen)
                    
                    is_toilet=np.isin(agent_loc, poss_toilets)
                    
                    if is_class==1:
                    
                        assigned_loc=1
                        
                    if is_canteen==1:
                        
                        assigned_loc=2
                        
                    if is_toilet==1:
                        
                        assigned_loc=3
                
#                print("is_non_corridor = ",is_non_corridor)
                
                #print("agent_loc = ",agent_loc)
                
                agent_output=sel_agent_info[time_step, selected_output]
                
                all_SAM2_output[sel_agent, assigned_loc]=all_SAM2_output[sel_agent, assigned_loc]+agent_output
                
                all_SAM2_count[sel_agent, assigned_loc]=all_SAM2_count[sel_agent, assigned_loc]+1
            
        print("All agent output SAM2")
        
        print(all_SAM2_output)
        
#        print("All agent output count")
        
 #       print(all_SAM2_count)
        
        all_scaled_SAM2_output=all_SAM2_output/all_SAM2_count
        
        max_measure=np.max(all_scaled_SAM2_output)
        
        min_measure=0
        
        nonzeros=np.nonzero(all_scaled_SAM2_output)
        
        if nonzeros[0].size!=0:
        
            min_measure=np.min(all_scaled_SAM2_output[np.nonzero(all_scaled_SAM2_output)])
        
        #min_measure=np.min(all_scaled_SAM2_output[np.nonzero(all_scaled_SAM2_output)])
        
        print("min_measure = ",min_measure)
        
        for sel_agent in np.arange(no_agents):
            
            for sel_loc in np.arange(4):
                
                all_scaled_SAM2_output_tmp=all_scaled_SAM2_output[sel_agent, sel_loc]
                
                if all_scaled_SAM2_output_tmp==0:
                    
                    all_scaled_SAM2_output[sel_agent, sel_loc]=min_measure+(max_measure-min_measure)/2
                    
        all_scaled_SAM2_output=np.transpose(all_scaled_SAM2_output)
        
        ##now we want to eliminate all locations where nothing ever happens
        
        total_situation_count=np.sum(all_SAM2_count,axis=1)
                
        fig, ax = plt.subplots()

        ax = sns.heatmap(all_scaled_SAM2_output, linewidth=0.5, cmap="coolwarm")

        ax.set_ylabel("Location ID")

        ax.set_xlabel("Agent")
        
        ##display it and save it
        
        plt.yticks([0.5, 1.5, 2.5, 3.5], ['Corridor', 'Class', 'Canteen', 'Toilet'], rotation=20)  # Set text labels and properties.

        plt.xticks([])

        plt.show()
        
        fig.savefig(f"heat_map_{selected_output_name}.png")
        
        plt.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        