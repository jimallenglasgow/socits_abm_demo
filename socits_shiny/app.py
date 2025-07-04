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

from general_functions import Update_Location_Graph

from general_functions import Generate_Oblong_Location_Grid
from general_functions import Generate_Oblong_Graph
from general_functions import Generate_Corridor
from general_functions import Generate_Floor_Corridor
from general_functions import Generate_Two_Floor_School
from general_functions import Generate_Two_Floor_School_Shut_Top_Floor
from general_functions import Generate_Two_Floor_School_Shut_Bottom_Floor

from general_functions import Assign_Groups

from general_functions import Set_New_Group_Goals

#from plot_output_functions import Plot_Output_Shiny

from run_single_model_function import Run_The_Model_Once

#from shiny.express import input, render
from shiny import reactive, Inputs, ui, App, render



#with ui.sidebar():
 
app_ui = ui.page_sidebar(

    ui.sidebar(
    
    ui.input_numeric("sel_time_step", "Selected time", 60, min=0, step=1),
    
    ui.input_radio_buttons("output_type", "Output of interest", choices=["Stress", "Loneliness"]),
    
    ui.input_radio_buttons("plot_type", "Plot type", choices=["Map", "Time", "Situation map"]),
    
        ui.accordion(
            ui.accordion_panel(
                "Initial settings",

                ui.input_slider("no_time_steps", "No. time steps", 10, 1000, 100, step = 1),
                ui.input_slider("no_students", "No. students", 10, 1000, 150, step = 1),

                ui.input_slider("stress_decay", "Stress decay", 0, 1, 0.1),

                


            ),
                
            ui.accordion_panel(
                "Background features",
                
                ##background features
                
                ui.input_slider("mean_time_stress", "Mean time stress", -1, 5, 1),
                ui.input_slider("mean_room_stress", "Mean room stress", -1, 5, 1),

    
                
            ),
            
            ui.accordion_panel(
                "Stress triggers",
                
                ##triggers
                
                ui.input_slider("crowded_threshold", "Crowded threshold", 0, 10, 2, step = 1),
                
                ui.input_slider("teacher_standards_mean", "Teacher standards mean", 0, 5, 3),

                ui.input_slider("student_standards_mean", "Student standards mean", 0, 5, 1),


                
            ),

            ui.accordion_panel(
                "Consequences for stress",
                
                ##consequences

                ui.input_slider("crowded_stress", "Crowded stress", 0, 10, 1.5),
                
                ui.input_slider("stress_through_class_interaction", "Stress through teacher int.", 0, 10, 1),
                
            ),
              
            
            ui.accordion_panel(
                "Student travel pars",
                
                ui.input_slider("prob_follow_group", "Prob follow group", 0, 1, 0.5),


                
            ),

            
            
                
        ),
            
 
    
    ),
    
    ui.div(
        ui.input_action_button(
            "run", "Update model output", class_="btn-primary"
        ),
 
    ),
    
    ui.output_plot("Plot_Model_Output"),
    
)



def server(input, output, session):
    
    ##some general inputs
    
    map_type="school"
    
    plot_type="0"
    
    full_emp_network=0

    all_agent_emp_inputs=0

    emp_SAM2=0

    no_emp_students=0
    
    use_emp_data=0

    display_some_outputs=1
    
    ##and the set inputs

    inc_science_perspective=1
    inc_walking_perspective=1
    inc_yais_perspective=1
    inc_teacher_perspective=1
    stress_decay=0.1
    status_threshold=5
    increase_in_stress_due_to_neg_int=2
    decrease_in_stress_due_to_pos_int=0.1
    rq_decrease_through_bullying=5e-04
    rq_increase_through_support=5e-04
    crowded_threshold=2
    crowded_stress=1.5
    journey_stress=5e-04
    stress_bully_scale=1
    stress_through_class_interaction=1
    prob_teacher_moving=0.5
    status_increase=5e-04
    status_decrease=5e-04
    mean_time_stress=1
    mean_room_stress=1
    prob_follow_group=0.5
    teacher_standards_mean=3.005
    teacher_standards_sd=0.5
    student_standards_mean=1
    student_standards_sd=0.5
    standards_interaction_scale=1
    initial_status_mean=2.5
    initial_status_sd=5
    reduction_due_to_teacher_presence=0.5
    rq_mean=0
    rq_sd=1
    initial_stress_scale=1
    stair_case_width=5
    corridor_width=2
    classroom_size=2
    no_classrooms_per_corridor=5
    floor_length=10
    toilets_loc_1=1
    toilets_loc_2=15
    canteen_loc=13
    staffroom_loc=6
    no_time_steps=60
    no_students=300
    no_bullies=0
    no_initially_stressed_students=0
    no_boys_in_each_age=100
    no_girls_in_each_age=100
    class_length=40
    toilet_prob=0.002
    canteen_prob=0.5



    non_classrooms=[toilets_loc_1, toilets_loc_2, canteen_loc, staffroom_loc]
    
    floor_width=no_classrooms_per_corridor*classroom_size+(no_classrooms_per_corridor-1)
    
    @reactive.calc()
    def Run_The_Shiny_Model():
        
        no_time_steps=input.no_time_steps()
        no_students=input.no_students()
        stress_decay=input.stress_decay()
        crowded_threshold=input.crowded_threshold()
        crowded_stress=input.crowded_stress()
        mean_time_stress=input.mean_time_stress()
        mean_room_stress=input.mean_room_stress()
        prob_follow_group=input.prob_follow_group()
        teacher_standards_mean=input.teacher_standards_mean()
        student_standards_mean=input.student_standards_mean()
        stress_through_class_interaction=input.stress_through_class_interaction()
        
        selected_students=np.arange(no_students)

        ########################################################

        ##run the model

        #############

        ##generate other required parameters
        
        no_teachers=(no_classrooms_per_corridor-1)*4+2

        no_age_groups=1#len(no_boys_in_each_age)

        no_agents=no_students+no_teachers

        no_groups=int(no_students/4)

        moving_times=np.arange(10, no_time_steps, class_length).astype(int)#[5, 25, 50, 75]
        
        lunch_time_ids=np.zeros(2)
        
        lunch_times=np.zeros(len(moving_times))#[0, 0, 0, 0]
        
        if no_time_steps>(class_length+10):
        
            lunch_1=int(len(moving_times)/2)
            lunch_2=int(len(moving_times)/2)+1
        
            lunch_time_ids[0]=lunch_1
            lunch_time_ids[1]=lunch_2    

            lunch_times[int(lunch_time_ids[0])]=1
            
            if lunch_2<len(moving_times):
            
                lunch_times[int(lunch_time_ids[1])]=1
            #lunch_times[7]=1
            
            
            
        
        non_classrooms=[toilets_loc_1, toilets_loc_2, canteen_loc, staffroom_loc]
        
        floor_width=no_classrooms_per_corridor*classroom_size+(no_classrooms_per_corridor-1)
        
        initial_rq_inputs=[rq_mean, rq_sd]

        initial_status_inputs=[initial_status_mean, initial_status_sd]
            
        student_teacher_pars=[teacher_standards_mean, teacher_standards_sd, student_standards_mean, student_standards_sd]

        #######################################################################################
        
        ##generate the input data based on the actual data
        
        sel_rows=selected_students

        emp_networks=0

        agent_emp_inputs=0

        if use_emp_data==1:
            
            agent_emp_inputs=all_agent_emp_inputs.iloc[sel_rows, ]

            sel_row_cols=selected_students-1

            emp_networks=full_emp_network.iloc[sel_row_cols, sel_row_cols]
            
            print("Sel emp network")

            print(emp_network)
        
        
        
        
        
        
        #######################################################################################

        ##Part xxx: initialise the locations

        grid_size=0
        
        if map_type=="grid":

            grid_size=floor_width*floor_length ##size of the grid 
            
            no_locations=floor_width*floor_length ##one location for each grid point

        if map_type=="school":

            grid_width=2*floor_width+stair_case_width ##width of the grid
            
            grid_length=floor_length ##length of the grid

            no_locations=grid_width*grid_length ##one location for each grid point

        all_locations=[] ##initialise the locations
        
    #        if input.update_geography()=="1":
        
        sel_location=0
        
        for x_coord in np.arange(grid_width):
            
            for y_coord in np.arange(grid_length):
            
                all_locations.append(Location(sel_location, [x_coord, y_coord], no_time_steps))
        
                sel_location=sel_location+1
        
    #    for sel_location in np.arange(no_locations): ##and then initialise them
        
     #       all_locations.append(Location(sel_location, floor_width*2+stair_case_width, floor_length, no_time_steps))
     
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
            
            G=Update_Location_Graph(all_locations, corridors_open, floor_width, stair_case_width, corridor_width, classroom_size, no_classrooms_per_corridor, non_classrooms, floor_length, display_some_outputs)
                    
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
        
        sel_canteen_teacher=np.random.permutation(np.arange(no_teachers))[0]

        #print("initial_agent_type")

        #print(initial_agent_types)#
        
        #########
        
        ##assign ages to each agent
        
        age_category=np.zeros(no_agents)
        
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

            all_agents.append(Agent(sel_agent, no_time_steps, age_category, initial_status_inputs, agent_emp_inputs, initial_stress_scale, no_teachers, use_emp_data))

        for sel_agent in np.arange(no_agents):

            all_agents[sel_agent].Record_All_Location_Types(all_locations) ##which are the classrooms

            all_agents[sel_agent].Initialise_Agent_Type(initial_agent_types, sel_canteen_teacher) ##what type are they
            
            all_agents[sel_agent].Initialise_Agent_Class(student_teacher_pars) ##what class are they?

        ##permute all the classrooms so that we can assign them to teachers

        poss_classrooms=all_agents[0].all_classrooms

    #    print("poss_classrooms")
        
     #   print(poss_classrooms)

        assigned_teacher_classrooms=np.random.permutation(poss_classrooms)

      #  print("assigned_teacher_classrooms")
        
       # print(assigned_teacher_classrooms)

        for sel_agent in np.arange(no_agents):

            all_agents[sel_agent].Initialise_RQ(all_agents, emp_networks, initial_agent_types, no_teachers, initial_rq_inputs) ##initialise the social network
            
        S2_agent_ids=np.where(age_category==1)[0]
        
        S4_agent_ids=np.where(age_category==2)[0]
        
     #   print("S4_agents")
        
      #  print(S4_agent_ids)
            
        S2_agents=[]
            
        for i in S2_agent_ids:
            
                S2_agents.append(all_agents[i])
                
        S4_agents=[]
            
        for i in S4_agent_ids:
            
                S4_agents.append(all_agents[i])
            
        assigned_groups=Assign_Groups(S2_agents, int(no_groups/2))
        
        assigned_groups=Assign_Groups(S4_agents, int(no_groups/2))

    #    print("assigned_groups")

     #   print(assigned_groups)
        
        for i in S4_agent_ids:
            
                all_agents[i].movement_group=all_agents[i].movement_group+int(no_groups/2)

    #    print("Example groups")
        
     #   print(all_agents[50].movement_group)
        
      #  print(all_agents[250].movement_group)

        Set_New_Group_Goals(no_groups, canteen_prob, 0, all_agents, no_teachers, no_in_each_age, all_locations)

        for sel_agent in np.arange(no_agents):

            all_agents[sel_agent].Initialise_Location(all_locations,assigned_teacher_classrooms, prob_follow_group) ##where do they start?

            all_agents[sel_agent].Initialise_Agent_Location_Time_Stress(all_locations,no_time_steps, mean_time_stress, mean_room_stress, inc_walking_perspective, inc_yais_perspective) ##how stressed are they in their initial location

            all_agents[sel_agent].Record_Agent_Info(0, all_locations) ##save this initial information

            
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

            all_agents[sel_agent].Set_New_Goal_Class(prob_teacher_moving, canteen_prob, 0, prob_follow_group, assigned_teacher_classrooms, all_locations) ##where is the new goal?

         #   all_agents[sel_agent].Ideal_Goal_Length(all_locations,G) ##how long should it take?

        #    print("goal length = ",all_agents[sel_agent].ideal_goal_length)

            ################

            ##stress functions

    #        all_agents[sel_agent].Decide_If_Stressed_Due_To_Time(all_locations,status_threshold,0)

     #       all_agents[sel_agent].Decide_If_Stressed_Due_To_Location(all_locations,status_threshold,0)


            ################

            ##record the info

            all_agents[sel_agent].Record_Agent_Info(0, all_locations)


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
                    
                    sel_stressed_students=all_students[0:(int(no_initially_stressed_students))]
                    
    #                print("sel_stressed_students")
                    
     #               print(sel_stressed_students)
     
                    for sel_agent in sel_stressed_students:
                        
                        all_agents[sel_agent].stress=3

            current_moving_time=moving_times[sel_moving_time]
            
            current_lunch_time=lunch_times[sel_moving_time] ##is it currently lunch time?
            
  #          print("current_lunch_time = ", current_lunch_time)
            
            if display_some_outputs==1:
            
                print("%age done = ",np.round(time/no_time_steps*100),"%", end="\r")

            ################

            ##movement functions

            ##and run through the actions of the agent
            
            ##decide if an agent should go to the toilet
                
            for sel_agent in np.arange(no_agents):

                agent_type=all_agents[sel_agent].agent_type ##agent type
                
                if agent_type==0:
                
                    r=np.random.random()
                    
                    if r<toilet_prob: ##...and by chance they don't move
                        
                        #print("Toilet!")
                        
                        all_agents[sel_agent].Set_New_Goal_Toilet(all_locations)

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

                Set_New_Group_Goals(no_groups, canteen_prob, current_lunch_time, all_agents, no_teachers, no_in_each_age, all_locations)

                for sel_agent in np.arange(no_agents):

                    all_agents[sel_agent].Set_New_Goal_Class(prob_teacher_moving, canteen_prob, current_lunch_time, prob_follow_group, assigned_teacher_classrooms, all_locations)

             #       all_agents[sel_agent].Ideal_Goal_Length(all_locations,G)



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

                all_agents[sel_agent].Decide_Negative_Interaction(all_agents, all_locations, status_threshold, stress_bully_scale)

                all_agents[sel_agent].Decide_Negative_Class_Interaction(all_agents, all_locations, standards_interaction_scale)

            ##################

            ##positive interactions (possibly in response to bullying)

            for sel_agent in np.arange(no_agents):

                all_agents[sel_agent].Decide_Positive_Interaction(all_agents,all_locations)

            ################

            ##update stress, rq and status based on which perspectives are included
            
            if inc_science_perspective==1:

                for sel_agent in np.arange(no_agents):

                    all_agents[sel_agent].Update_Stress_Due_To_Interactions(all_locations, all_agents, increase_in_stress_due_to_neg_int, decrease_in_stress_due_to_pos_int, reduction_due_to_teacher_presence)

             #       all_agents[sel_agent].Update_Status(status_increase,status_decrease)

                    all_agents[sel_agent].Update_Loneliness_Calc()

            #        all_agents[sel_agent].Update_All_RQ_Based_On_Interaction(rq_decrease_through_bullying,rq_increase_through_support)

            if inc_walking_perspective==1 or inc_yais_perspective==1:

                for sel_agent in np.arange(no_agents):
                    
                    all_agents[sel_agent].Decide_If_Stressed_Due_To_Time(all_locations,status_threshold,time)

                    all_agents[sel_agent].Decide_If_Stressed_Due_To_Location(all_locations,status_threshold,time)

                    all_agents[sel_agent].Decide_If_Stressed_Due_To_Crowdedness(all_locations,crowded_threshold,crowded_stress)
                    
#            if inc_yais_perspective==1:

 #               for sel_agent in np.arange(no_agents):

  #                  #print("Journey time = ",all_agents[sel_agent].journey_time,", Ideal journey time = ",all_agents[sel_agent].ideal_goal_length)
                    
   #                 all_agents[sel_agent].Decide_If_Stressed_Due_To_Delay(journey_stress)

                
            if inc_teacher_perspective==1:
            
                for sel_agent in np.arange(no_agents):

                    all_agents[sel_agent].Update_Stress_Due_To_Class_Interactions(stress_through_class_interaction, decrease_in_stress_due_to_pos_int)
                
            for sel_agent in np.arange(no_agents):
                
                all_agents[sel_agent].Decay_Stress(stress_decay, time)
                
                all_agents[sel_agent].Calc_Network_Degree()
                
                all_agents[sel_agent].Calc_Situational_Loneliness(all_locations)

            ################

            ##record the info

            for sel_agent in np.arange(no_agents):

                all_agents[sel_agent].Record_Agent_Info(time, all_locations)
                    
            all_outputs=[all_agents, all_locations]
        
        return all_outputs
    

    @render.plot
    # ignore_none=False is used to instruct Shiny to render this plot even before the
    # input.run button is clicked for the first time. We do this because we want to
    # render the empty 3D space on app startup, to give the user a sense of what's about
    # to happen when they run the simulation.
    @reactive.event(input.run, ignore_none=True)

    def Plot_Model_Output():
        
        sel_time_step=input.sel_time_step()
        
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

#        selected_output_name="Stress"
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
            
            mean_stress=np.mean(all_agent_output[:,student_agents])
            
            print("Mean stress = ",mean_stress)

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
                
            ax.plot(np.arange(no_time_steps), np.ones(no_time_steps)*mean_stress, linestyle="dashed", linewidth=5)

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
                    
                    agent_loc=int(sel_agent_info[time_step, 9])#-1
                    
          #          print("agent_loc = ",agent_loc)
                    
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


