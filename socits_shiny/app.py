##shiny run --reload --launch-browser GitHub/socits_abm_demo/socits_shiny/app.py

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

import seaborn as sns

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

#from shiny.express import input, render
from shiny import reactive, Inputs, ui, App, render



#with ui.sidebar():
 
app_ui = ui.page_sidebar(

    ui.sidebar(
    
    ui.input_slider("sel_time_step", "Selected time", 0, 300, 50, step=1),
    
    ui.input_radio_buttons("output_type", "Output of interest", choices=["Stress", "Loneliness"]),
    
    ui.input_radio_buttons("plot_type", "Plot type", choices=["Map", "Time", "Situation Map"]),
    
        ui.accordion(
            ui.accordion_panel(
                "Initial settings",

                ui.input_slider("no_students", "No. students", 1, 300, 100, step=1),

                ui.input_slider("no_teachers", "No. teachers", 1, 50, 10, step=1),
                
                ui.input_slider("stress_decay_level", "Stress decay rate", 0, 1, 0.2),

                ui.input_slider("prob_teacher_moving", "Prob. teacher moving", 0, 1, 0.5),

                ui.input_slider("toilet_prob", "Prob. being in the toilet", 0, 1, 0.4),

                ui.input_slider("canteen_prob", "Prob. eating lunch in the canteen", 0, 1, 0.8),


            ),
                
            ui.accordion_panel(
                "Background features",
                
                ##background features
    
                ui.input_slider("mean_room_stress", "Mean room stress", 0, 5, 0),
                
                ui.input_slider("mean_time_stress", "Mean time stress", 0, 5, 0),
                
            ),
            
            ui.accordion_panel(
                "Stress triggers",
                
                ##triggers

                ui.input_slider("status_threshold", "Status threshold", 0, 5, 1),

                ui.input_slider("stress_bully_scale", "Stress bully scale", 0, 5, 1),

                ui.input_slider("crowded_threshold", "No. others needed for crowding stress", 0, 10, 2, step=1),
                
            ),

            ui.accordion_panel(
                "Consequences for stress",
                
                ##consequences

                ui.input_slider("increase_in_stress_due_to_neg_int", "Stress inc. through neg. int.", 0, 5, 1.2),

                ui.input_slider("decrease_in_stress_due_to_pos_int", "Stress dec. through pos. int.", 0, 5, 0.2),

                ui.input_slider("status_increase", "Status inc. through neg. int.", 0, 5, 1),

                ui.input_slider("status_decrease", "Status dec. through neg. int.", 0, 5, 1),

                ui.input_slider("crowded_stress", "Stress due to crowding", 0, 5, 1),

                ui.input_slider("journey_stress", "Stress due to lateness", 0, 5, 1),

                ui.input_slider("stress_through_class_interaction", "Stress through uniform int.", 0, 5, 1),
                
            ),
              
            
            ui.accordion_panel(
                "Student travel pars",
                
                ui.input_slider("no_groups", "No. friendship groups", 1, 50, 10, step=1),
    
                ui.input_slider("prob_follow_group", "Prob. follow friends", 0, 1, 0.9),

                ui.input_slider("move_with_friends", "Prob. move with friends", 0, 1, 0.5),
                
            ),

            
            
                
        ),
            
 
    
    ),
    
    ui.div(
        ui.input_action_button(
            "run", "Run simulation", class_="btn-primary"
        ),
 
    ),
    
    ui.output_plot("Plot_Model_Output"),
    
)



def server(input, output, session):
    
    map_type="school"

    floor_width=10

    floor_length=10

    stair_case_width=5

    corridor_width=1
    
    use_emp_networks=0
    
    ##time steps

    no_time_steps=200
    
    no_bullies=0
    
    moving_times=[10, 40, 60, 80, 110]

    lunch_times=[0, 1, 0, 1, 0]
    
    ##parameters to include certain perspective (1=include)

    inc_science_perspective=1
    
    inc_walking_perspective=1

    inc_yais_perspective=1

    inc_teacher_perspective=1
    
    rq_decrease_through_bullying=0 ##change in relationship quality due to a negative interaction

    rq_increase_through_support=0 ##change in relationship quality due to a positive interaction
    
    ##triggers

    no_classes=4 ##no. of different possible classes

    class_prob_dist=[0.25,0.25,0.25,0.25] ##how these classes are distributed in the population

    @reactive.calc()
    def Run_The_Shiny_Model():
        
        ##no. of different agent types

        no_students=input.no_students()

        no_teachers=input.no_teachers()

        no_agents=no_students+no_teachers

  #      no_groups=input.no_groups()

        ##parameters common to all perspectives

        stress_decay=input.stress_decay_level()
        
        no_groups=input.no_groups()

        prob_teacher_moving=input.prob_teacher_moving() ##the probability of a teacher moving around the school
            
        move_with_friends=input.move_with_friends() ##should individuals try to move with friends?  1==Yes, 0==No

        mean_time_stress=input.mean_time_stress() ##the mean of the normal distribution for the time stress

        mean_room_stress=input.mean_room_stress()
        
        toilet_prob=input.toilet_prob()#0.4 ##probability of any student going to the toilet at any one time

        canteen_prob=input.canteen_prob()#0.8 ##probability of any student eating lunch in the canteen

        prob_follow_group=input.prob_follow_group()#0.9

        ##science model

        ##triggers

        status_threshold=input.status_threshold()#1 ##theshold for the status difference before a negative interaction

        stress_bully_scale=input.stress_bully_scale()#1

        ##consequences

        increase_in_stress_due_to_neg_int=input.increase_in_stress_due_to_neg_int()#1.2 ##change in stress due to a negative interaction

        decrease_in_stress_due_to_pos_int=input.decrease_in_stress_due_to_pos_int()#0.2 ##change in stress due to a positive interaction

        status_increase=input.status_increase()#1 ##increase in status if an individual interacts negatively

        status_decrease=input.status_decrease()#1 ##decrease in status if an individual receives a negative interaction

        ##yai's model

        ##triggers

        crowded_threshold=input.crowded_threshold()#2 ##number of others in a location before stress increases

        ##consequences

        crowded_stress=input.crowded_stress()#1 ##the amount by which stress increases due to crowdedness

        journey_stress=input.journey_stress()#1 ##stress increase if a journey is disrupted

        ##walking interviews model

        ##No additional parameters required

        ##consequences

        stress_through_class_interaction=input.stress_through_class_interaction()#1 ##stress increase due to negative interaction between students and teachers

        ########################################################

        ##Part C: run the model

        ##the following line calls the function "Run_The_Model_Once" in the file "run_single_model_function.py" to run the model, and returns all of the agents information in each time step
            
        plot_type="0"
        
        all_technical_inputs=[plot_type, map_type, floor_width, floor_length, stair_case_width, corridor_width, no_time_steps, no_students, no_teachers, no_bullies,inc_science_perspective, inc_walking_perspective, inc_yais_perspective, inc_teacher_perspective, no_classes, moving_times, move_with_friends, lunch_times, no_groups, class_prob_dist, use_emp_networks]

        all_inputs_set_through_data=[toilet_prob, canteen_prob]

        all_calibrated_inputs=[stress_decay, status_threshold, increase_in_stress_due_to_neg_int, decrease_in_stress_due_to_pos_int, rq_decrease_through_bullying,rq_increase_through_support, crowded_threshold, crowded_stress, journey_stress, stress_bully_scale, stress_through_class_interaction, prob_teacher_moving, status_increase, status_decrease, mean_time_stress, mean_room_stress, prob_follow_group]

        print("Run the model")

        all_outputs=Run_The_Model_Once(all_calibrated_inputs, all_inputs_set_through_data, all_technical_inputs)
        
        return all_outputs
    

    @render.plot
    # ignore_none=False is used to instruct Shiny to render this plot even before the
    # input.run button is clicked for the first time. We do this because we want to
    # render the empty 3D space on app startup, to give the user a sense of what's about
    # to happen when they run the simulation.
    @reactive.event(input.run, ignore_none=False)

    def Plot_Model_Output():
        
        sel_time_step=input.sel_time_step()

        ##initialise the output plot

        fig, ax = plt.subplots()
        
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
        
        plt.title(f"Map of the school at time point {sel_time_step}", fontsize=20)

        plt.xticks([])
        plt.yticks([])
            
        if input.run() > 0:
            
            print("Now run the model")
            
            all_outputs=Run_The_Shiny_Model()
            
            all_agents=all_outputs[0]

            all_locations=all_outputs[1]

            ###########

            ##save the model output

            ##first generate an array which contains each agent's info at each timestep

            sel_agent=0
            
            no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

            no_agents=len(all_agents)

            model_output=np.hstack([np.zeros([no_time_steps,1]),all_agents[sel_agent].agent_info])

            for sel_agent in np.arange(1,no_agents):

                sel_agent_model_output=np.hstack([np.ones([no_time_steps,1])*sel_agent,all_agents[sel_agent].agent_info])

                model_output=np.vstack([model_output,sel_agent_model_output])


            selected_output_name="Stress"

            if input.output_type()=="Stress":

                selected_output=3
                
            elif input.output_type()=="Loneliness":

                selected_output=4
                
                selected_output_name="Loneliness"

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
                
                plt.title(f"Change over time for four students at {no_time_steps} time points in the school day", fontsize=20)
            
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

                ##initialise the output plot

                fig, ax = plt.subplots()

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
                
                plt.title(f"Map of the school at time point {sel_time_step}", fontsize=20)

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

            if plot_type=="Situation map": ##plot the SOCITS squares from the model
                
                no_agents=len(all_agents) ##find the number of agents
                
                no_locations=len(all_locations)

                no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

                all_SAM2_output=np.zeros([no_agents, 4]) ##initialise agent output for all agents
                
                all_SAM2_count=np.ones([no_agents, 4]) ##initialise how many time steps this data is for
                
                poss_classrooms=all_agents[0].all_classrooms
                
                poss_toilets=all_agents[0].all_toilets
                
                poss_canteen=all_agents[0].all_canteens
                
        #        print("poss_classrooms")
                
         #       print(poss_classrooms)
                
          #      print("poss_toilets")
                
           #     print(poss_toilets)
                
            #    print("poss_canteen")
                
             #   print(poss_canteen)
                
                all_non_corridors=np.hstack([poss_classrooms, poss_toilets, poss_canteen])
                
              #  print("all_non_corridors")
                
               # print(all_non_corridors)

                for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

                    sel_agent_info=all_agents[sel_agent].agent_info
                    
                    for time_step in np.arange(no_time_steps):
                        
                        agent_loc=int(sel_agent_info[time_step, 8])-1
                        
                #        print("agent_loc = ",agent_loc)
                        
                        assigned_loc=0
                        
                        is_non_corridor=np.isin(agent_loc, all_non_corridors)
                        
                        if is_non_corridor==1:
                            
                            is_class=np.isin(agent_loc, poss_classrooms)
                            
                            is_canteen=np.isin(agent_loc, poss_canteen)
                            
                            is_toilet=np.isin(agent_loc, poss_toilets)
                            
                            if is_class==1:
                            
                                assigned_loc=1
                                
                            if is_toilet==1:
                                
                                assigned_loc=3
                                
                            if is_canteen==1:
                                
                                assigned_loc=2
                                
                            
                        
        #                print("is_non_corridor = ",is_non_corridor)
                        
                        #print("agent_loc = ",agent_loc)
                        
                 #       print("assigned_loc = ",assigned_loc)
                        
                        agent_output=sel_agent_info[time_step, selected_output]
                        
                        all_SAM2_output[sel_agent, assigned_loc]=all_SAM2_output[sel_agent, assigned_loc]+agent_output
                        
                        all_SAM2_count[sel_agent, assigned_loc]=all_SAM2_count[sel_agent, assigned_loc]+1
                
                
        #        print("All agent output SAM2")
                
         #       print(all_SAM2_output)
                
                all_scaled_SAM2_output=all_SAM2_output/all_SAM2_count
                
                max_measure=np.max(all_scaled_SAM2_output)
                
                min_measure=0
                
                nonzeros=np.nonzero(all_scaled_SAM2_output)
                
                if nonzeros[0].size!=0:
                
                    min_measure=np.min(all_scaled_SAM2_output[np.nonzero(all_scaled_SAM2_output)])
                
          #      print("min_measure = ",min_measure)
                
                for sel_agent in np.arange(no_agents):
                    
                    for sel_loc in np.arange(4):
                        
                        all_scaled_SAM2_output_tmp=all_scaled_SAM2_output[sel_agent, sel_loc]
                        
                        if all_scaled_SAM2_output_tmp==0:
                            
                            all_scaled_SAM2_output[sel_agent, sel_loc]=min_measure+(max_measure-min_measure)/2
                            
                            

        #        print("All agent output output")
                
         #       print(all_scaled_SAM2_output)

                all_scaled_SAM2_output=np.transpose(all_scaled_SAM2_output)
                
                ##now we want to eliminate all locations where nothing ever happens
                
                total_situation_count=np.sum(all_SAM2_count,axis=1)
                        
                fig, ax = plt.subplots()

                ax = sns.heatmap(all_scaled_SAM2_output, linewidth=0.5, cmap="coolwarm")
                
                sns.color_palette("coolwarm", as_cmap=True)

                ax.set_ylabel("Location ID")

                ax.set_xlabel("Agent")
                
                plt.title("Measure in each situation", fontsize=20)
                
                ##display it and save it
                
                plt.yticks([0.5, 1.5, 2.5, 3.5], ['Corridor', 'Class', 'Canteen', 'Toilet'], rotation=20)  # Set text labels and properties.

                plt.xticks([])

        return fig
     
     
     
     
     
app = App(app_ui, server)


def Socits_Model():

    ##general variables

    ##run the model code?

    ##1=run the model before plotting

    ##0=load data from the last run
    
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

    no_time_steps=120

    ##no. of different agent types

    no_students=input.no_students()

    no_teachers=input.no_teachers()

    no_agents=no_students+no_teachers

    no_bullies=0
    
    no_groups=input.no_groups()

    ##parameters common to all perspectives

    stress_decay_level=input.stress_decay_level()

    prob_teacher_moving=input.prob_teacher_moving() ##the probability of a teacher moving around the school
        
    move_with_friends=input.move_with_friends() ##should individuals try to move with friends?  1==Yes, 0==No

    moving_times=[10, 40, 60, 80, 110]

    lunch_times=[0, 1, 0, 1, 0]

    mean_time_stress=input.mean_time_stress() ##the mean of the normal distribution for the time stress

    mean_room_stress=input.mean_room_stress()
    
    toilet_prob=input.toilet_prob()#0.4 ##probability of any student going to the toilet at any one time

    canteen_prob=input.canteen_prob()#0.8 ##probability of any student eating lunch in the canteen

    prob_follow_group=input.prob_follow_group()#0.9

    ##parameters to include certain perspective (1=include)

    inc_science_perspective=1
    
    inc_walking_perspective=1

    inc_yais_perspective=1

    inc_teacher_perspective=1

    ##science model

    ##triggers

    status_threshold=input.status_threshold()#1 ##theshold for the status difference before a negative interaction

    stress_bully_scale=input.stress_bully_scale()#1

    ##consequences

    increase_in_stress_due_to_neg_int=input.increase_in_stress_due_to_neg_int()#1.2 ##change in stress due to a negative interaction

    decrease_in_stress_due_to_pos_int=input.decrease_in_stress_due_to_pos_int()#0.2 ##change in stress due to a positive interaction

    rq_decrease_through_bullying=0 ##change in relationship quality due to a negative interaction

    rq_increase_through_support=0 ##change in relationship quality due to a positive interaction

    status_increase=input.status_increase()#1 ##increase in status if an individual interacts negatively

    status_decrease=input.status_decrease()#1 ##decrease in status if an individual receives a negative interaction

    ##yai's model

    ##triggers

    crowded_threshold=input.crowded_threshold()#2 ##number of others in a location before stress increases

    ##consequences

    crowded_stress=input.crowded_stress()#1 ##the amount by which stress increases due to crowdedness

    journey_stress=input.journey_stress()#1 ##stress increase if a journey is disrupted

    ##walking interviews model

    ##No additional parameters required

    ##teachers model

    ##triggers

    no_classes=4 ##no. of different possible classes

    class_prob_dist=[0.25,0.25,0.25,0.25] ##how these classes are distributed in the population

    ##consequences

    stress_through_class_interaction=input.stress_through_class_interaction()#1 ##stress increase due to negative interaction between students and teachers

    ########################################################

    ##Part C: run the model

    ##the following line calls the function "Run_The_Model_Once" in the file "run_single_model_function.py" to run the model, and returns all of the agents information in each time step
        
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


    selected_output_name="Stress"

    if input.output_type()=="Stress":

        selected_output=3
        
    elif input.output_type()=="Loneliness":

        selected_output=4
        
        selected_output_name="Loneliness"

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
        
        plt.title(f"Change over time for four students at {no_time_steps} time points in the school day", fontsize=20)
    
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

        ##initialise the output plot

        fig, ax = plt.subplots()

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
        
        plt.title(f"Map of the school at time point {sel_time_step}", fontsize=20)

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

    if plot_type=="Situation map": ##plot the SOCITS squares from the model
        
        no_agents=len(all_agents) ##find the number of agents
        
        no_locations=len(all_locations)

        no_time_steps=len(all_agents[0].agent_info[:,0]) ##and the number of time steps

        all_SAM2_output=np.zeros([no_agents, 4]) ##initialise agent output for all agents
        
        all_SAM2_count=np.ones([no_agents, 4]) ##initialise how many time steps this data is for
        
        poss_classrooms=all_agents[0].all_classrooms
        
        poss_toilets=all_agents[0].all_toilets
        
        poss_canteen=all_agents[0].all_canteens
        
#        print("poss_classrooms")
        
 #       print(poss_classrooms)
        
  #      print("poss_toilets")
        
   #     print(poss_toilets)
        
    #    print("poss_canteen")
        
     #   print(poss_canteen)
        
        all_non_corridors=np.hstack([poss_classrooms, poss_toilets, poss_canteen])
        
      #  print("all_non_corridors")
        
       # print(all_non_corridors)

        for sel_agent in np.arange(no_agents): ##for each agent, assign the specific output to the output array

            sel_agent_info=all_agents[sel_agent].agent_info
            
            for time_step in np.arange(no_time_steps):
                
                agent_loc=int(sel_agent_info[time_step, 8])-1
                
        #        print("agent_loc = ",agent_loc)
                
                assigned_loc=0
                
                is_non_corridor=np.isin(agent_loc, all_non_corridors)
                
                if is_non_corridor==1:
                    
                    is_class=np.isin(agent_loc, poss_classrooms)
                    
                    is_canteen=np.isin(agent_loc, poss_canteen)
                    
                    is_toilet=np.isin(agent_loc, poss_toilets)
                    
                    if is_class==1:
                    
                        assigned_loc=1
                        
                    if is_toilet==1:
                        
                        assigned_loc=3
                        
                    if is_canteen==1:
                        
                        assigned_loc=2
                        
                    
                
#                print("is_non_corridor = ",is_non_corridor)
                
                #print("agent_loc = ",agent_loc)
                
         #       print("assigned_loc = ",assigned_loc)
                
                agent_output=sel_agent_info[time_step, selected_output]
                
                all_SAM2_output[sel_agent, assigned_loc]=all_SAM2_output[sel_agent, assigned_loc]+agent_output
                
                all_SAM2_count[sel_agent, assigned_loc]=all_SAM2_count[sel_agent, assigned_loc]+1
        
        
#        print("All agent output SAM2")
        
 #       print(all_SAM2_output)
        
        all_scaled_SAM2_output=all_SAM2_output/all_SAM2_count
        
        max_measure=np.max(all_scaled_SAM2_output)
        
        min_measure=0
        
        nonzeros=np.nonzero(all_scaled_SAM2_output)
        
        if nonzeros[0].size!=0:
        
            min_measure=np.min(all_scaled_SAM2_output[np.nonzero(all_scaled_SAM2_output)])
        
  #      print("min_measure = ",min_measure)
        
        for sel_agent in np.arange(no_agents):
            
            for sel_loc in np.arange(4):
                
                all_scaled_SAM2_output_tmp=all_scaled_SAM2_output[sel_agent, sel_loc]
                
                if all_scaled_SAM2_output_tmp==0:
                    
                    all_scaled_SAM2_output[sel_agent, sel_loc]=min_measure+(max_measure-min_measure)/2
                    
                    

#        print("All agent output output")
        
 #       print(all_scaled_SAM2_output)

        all_scaled_SAM2_output=np.transpose(all_scaled_SAM2_output)
        
        ##now we want to eliminate all locations where nothing ever happens
        
        total_situation_count=np.sum(all_SAM2_count,axis=1)
                
        fig, ax = plt.subplots()

        ax = sns.heatmap(all_scaled_SAM2_output, linewidth=0.5, cmap="coolwarm")
        
        sns.color_palette("coolwarm", as_cmap=True)

        ax.set_ylabel("Location ID")

        ax.set_xlabel("Agent")
        
        plt.title("Measure in each situation", fontsize=20)
        
        ##display it and save it
        
        plt.yticks([0.5, 1.5, 2.5, 3.5], ['Corridor', 'Class', 'Canteen', 'Toilet'], rotation=20)  # Set text labels and properties.

        plt.xticks([])

#        plt.show()
        
 #       plt.close()
 
    return fig
 
