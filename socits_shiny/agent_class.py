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

##the following code defines the agent class, and all functions included involving the agent (calculating stress etc.)

class Agent():

    ##this function initialises all the values for the agent

    def __init__(self,agent_id, no_time_steps, age_category, initial_status_inputs, agent_demos, initial_stress_scale, no_teachers, use_emp_data):

        self.agent_id=agent_id
        
        self.agent_info=np.zeros([no_time_steps,9])
        
        ####
        
        ##initialise the age
        
        self.age=1
        
        if use_emp_data==1:
        
            age_category=np.array(agent_demos.age)
        
            if agent_id>no_teachers:
            
                self.age=age_category[agent_id-no_teachers]
            
            
        #print("age = ", self.age)
        
        ####
        
        ##location/movement values
        
        self.current_position=0

        self.current_location=0
        
#        self.goal_j=np.random.permutation([0,9])[0]

 #       self.goal_i=np.random.randint(10)

  #      self.goal_coord=[self.goal_i,self.goal_j]

        self.goal_location=0#10*self.goal_i+self.goal_j
        
        self.journey_time=0
        
        self.go_to_toilet=0
        
        self.movement_group=0
        
        self.group_goals=0
        
        self.teacher_classroom=-1
        
        self.canteen_duty=0
        
        #####
        
        ##stress/loneliness values
       
        self.agent_class=0

        self.ideal_goal_length=0
        
        self.stress=0
        
        raw_initial_agent_stress=np.random.random()*10
        
        self.stress=raw_initial_agent_stress*initial_stress_scale
        
        if self.stress<0:
            
            self.stress=0
        
        if use_emp_data==1:
        
            raw_initial_agent_stress=np.array(agent_demos.Initial_Stress)
        
            initial_agent_stress=raw_initial_agent_stress*initial_stress_scale
            
            if agent_id>no_teachers:
            
                self.stress=initial_agent_stress[agent_id-no_teachers]

        #print("initial stress = ", self.stress)

        self.base_stress=0

        self.location_stress=0

        self.time_stress=0
        
        self.status=np.random.normal(loc=initial_status_inputs[0], scale=initial_status_inputs[1])
        
        if self.status<0:
            
            self.status=0
        
        self.pos_deg=0
        
        self.neg_deg=0
        
        self.location_loneliness=0
        
        ####

        ##interaction variables

        self.neg_int_prov=0

        self.neg_int_rec=0

        self.bullying=-1

        self.bullied_by=-1

        self.pos_int_prov=0

        self.pos_int_rec=0

        self.supporting=-1

        self.supported_by=-1
        
        self.neg_class_int_prov=0

        self.neg_class_int_rec=0

        self.bullied_by_class=-1

        ####

        ##initial loneliness variables

        self.rq=0

        self.all_rq=0
        
        ######
        
        ##initial location variables
        
        self.available_locations=0
        
        self.all_classrooms=0
        
        self.all_toilets=0
        
        self.all_canteens=0
        
        self.all_staffrooms=0
        
    ##################################
    
    ##this function tells the agent which locations are classrooms
    
    ##Function: Record_All_Classrooms
    
    ##Inputs: all locations
    
    ##Outputs: A vector of all locations which are classrooms
    
    ##What it does
    
    ##This function looks at each location in turn, and if it is a classroom adds this to a vector for the agent to refer to each time they set a new goal.
        
    def Record_All_Location_Types(self,all_locations):
    
        no_locations=len(all_locations)
        
        all_poss_locations=[]

        poss_classrooms=[] ##initialise an empty classroom vector
        
        poss_toilets=[]
        
        poss_canteen=[]
        
        poss_staffroom=[]
        
  #      print(no_locations)
        
        for sel_location in np.arange(no_locations): ##checking through each location....

 #           print("Sel location = ",sel_location)

            available_location=all_locations[sel_location].possible_location
            
            sel_room_id=all_locations[sel_location].room_id
            
            if available_location==1: ##if the location is a classroom, add it to the vector
            
                all_poss_locations=np.hstack([all_poss_locations,sel_room_id])


            ###

            classroom_location=all_locations[sel_location].is_classroom
            
            if classroom_location==1: ##if the location is a classroom, add it to the vector
            
                agent_year=self.age
            
                if agent_year==0:
                
                    poss_classrooms=np.hstack([poss_classrooms,sel_room_id])
            
                else:
            
                    room_year=all_locations[sel_location].classroom_year
                
                    if room_year==agent_year:
                
                        poss_classrooms=np.hstack([poss_classrooms,sel_room_id])
                
            ###
                
            toilet_location=all_locations[sel_location].is_toilet
            
            if toilet_location==1: ##if the location is a classroom, add it to the vector
            
                poss_toilets=np.hstack([poss_toilets,sel_room_id])
            
            ####
            
            canteen_location=all_locations[sel_location].is_canteen
            
            if canteen_location==1: ##if the location is a classroom, add it to the vector
            
                poss_canteen=np.hstack([poss_canteen,sel_room_id])
                
            ####
                
            staffroom_location=all_locations[sel_location].is_staff_room
            
            if staffroom_location==1: ##if the location is a classroom, add it to the vector
                
#                print("Is staffroom")
            
                poss_staffroom=np.hstack([poss_staffroom,sel_room_id])

        self.all_classrooms=poss_classrooms ##and save the vector
        
        self.all_toilets=poss_toilets ##and save the vector
        
        self.all_canteens=poss_canteen ##and save the vector
        
        self.all_staffrooms=poss_staffroom ##and save the vector

        self.available_locations=all_poss_locations

#        print("self.all_classrooms")

 #       print(self.all_classrooms)
        
  #      print("self.all_toilets")
        
   #     print(self.all_toilets)
        
    #    print("self.all_canteens")
        
     #   print(self.all_canteens)
        
#        print("self.all_staffrooms")
 #       print(self.all_staffrooms)
        

    ##################################
    
    ##this function initialises the location of the agent
    
    ##Function: Initialise_Location
    
    ##Inputs: all locations; the classrooms assigned to each teacher
    
    ##Outputs: The initial location of our agent
    
    ##What it does
    
    ##If the agent is a student, then they are randomly assigned to a classroom.  If they are a teacher, they are allocated to a classroom based on their agent ID, so that teachers are spread between classrooms.

    def Initialise_Location(self, all_locations, assigned_teacher_classrooms, prob_follow_group):

#        no_locations=len(all_locations)

        poss_initial_locations=self.all_classrooms ##the possible initial locations are classrooms

        initial_location_id=int(np.random.permutation(poss_initial_locations)[0]) ##select one of the classrooms at random
        
        initial_location=self.Select_Random_Room_Location(initial_location_id, all_locations)
        
        r=np.random.random()

        if r<prob_follow_group:
            
            own_group=self.movement_group
            
            initial_location=int(self.group_goals[own_group])
        
        ##place any teacher into a single classroom
        
        agent_type=self.agent_type
        
        if agent_type==2: ##if the agent is a teacher
        
            no_classrooms=len(assigned_teacher_classrooms) ##check the number of classrooms
        
            classroom_id=int(np.mod(self.agent_id, no_classrooms))+1 ##and use the agent ID to work out which class to put the teacher in
            
            initial_location=self.Select_Random_Room_Location(classroom_id, all_locations)
            
            #initial_location=int(assigned_teacher_classrooms[classroom_id]) ##assign this as the initial location
            
            self.goal_location=initial_location ##and also set the goal to the current location for the teachers that don't move
            
            self.teacher_classroom=initial_location
            
        self.current_location=initial_location ##and set the current location

        current_location_coords=all_locations[initial_location].coords ##set the coordinates...

        new_agent_coords_shift=np.random.random(2)-0.5  ##...and shift them a little so that not everyone is in the same place

        new_agent_coords=new_agent_coords_shift+current_location_coords

#        print("current_agent coords = ",new_agent_coords)

        self.current_position=new_agent_coords

    ##################################
    
    ##this function just records the type (set globally when initialising the variables)
    
    ##Function: Initialise_Agent_Type
    
    ##Inputs: Vector of agent types
    
    ##Outputs: The individual agents type
    
    ##What it does
    
    ##A simple function to assign the agent type

    def Initialise_Agent_Type(self,initial_agent_types, sel_canteen_teacher):

        self.agent_type=initial_agent_types[self.agent_id]
        
        if self.agent_id==sel_canteen_teacher:
            
            self.canteen_duty=1
            
            

    ##################################
    
    ##this function randomly assigns the agent a class
    
    ##Function: Initialise_Agent_Class
    
    ##Inputs: The number of classes, and the probability of each
    
    ##Outputs: The selected agent's class
    
    ##What it does
    
    ##Randomly assigns the agent a class based on the probability distribution of the classes

    def Initialise_Agent_Class(self, student_teacher_pars):
        
        agent_type=self.agent_type
        
        class_mean=student_teacher_pars[2]
        
        class_sd=student_teacher_pars[3]
        
        if agent_type==2:
            
            class_mean=student_teacher_pars[0]
        
            class_sd=student_teacher_pars[1]
    
#        sel_class=np.random.choice(no_classes, 1, p=class_prob_dist)

        sel_class=np.random.normal(loc=class_mean, scale=class_sd)
        
        if sel_class<0:
            
            sel_class=0
            
        self.agent_class=sel_class
        
#        print("sel_class = ",self.agent_class)

    ##################################
    
    ##this function sets which times and locations are stressful for this agent
    
    ##Function: Initialise_Agent_Location_Time_Stress
    
    ##Inputs: All locations, the total number of time steps
    
    ##Outputs: A baseline stress for each time and location for this specific agent
    
    ##What it does
    
    ##This assigns a random stress to each classroom for the agent, to match the fact that some classrooms are more or less nice.  The function also does the same for specified time steps, in particular those in the middle 10% and the last 20% of the day.
    
    def Initialise_Agent_Location_Time_Stress(self,all_locations,no_time_steps, mean_time_stress, mean_room_stress, inc_walking_perspective, inc_yais_perspective):

        if self.agent_type!=2: ##if the agent is a student

            no_locations=len(all_locations)

            self.location_stress=np.zeros([no_locations]) ##initialise a vector of stress for each location

            self.time_stress=np.zeros([no_time_steps]) ##and initialise a vector of stress for each time step

            ind_time_stress=mean_time_stress#np.random.normal(loc=mean_time_stress)#*0.05 ##generate a constant random number for how stressful stressful times are for this agent
            
            if ind_time_stress<0.01:
            
                ind_time_stress=0
                
            if mean_time_stress<0.01:
                
                ind_time_stress=0
                
            if inc_walking_perspective==1 or inc_yais_perspective==1:

                for time in np.arange(no_time_steps): ##assign the correct stress for each time step

    #                if time/no_time_steps>0.8: ##if the time is above 80% of the time, individuals find it stressful

                        self.time_stress[time]=ind_time_stress
                        
     #               if time/no_time_steps>0.45 and time/no_time_steps<0.55: ##and if it's between 45% and 55%

      #                  self.time_stress[time]=ind_time_stress
                        
                poss_classrooms=self.all_classrooms

                for sel_location in poss_classrooms: ##for each classroom, randomly assign a stress
                    
                    sel_room_stress=np.random.normal(loc=mean_room_stress)
                    
                    if sel_room_stress<0.01:
                    
                        sel_room_stress=0
                        
                    if mean_room_stress<0.01:
                        
                        sel_room_stress=0
                    
                    self.location_stress[int(sel_location)]=sel_room_stress
                    

    ###############################################################

    ##function to select a location at random for the agent to move to

    def Select_Random_Room_Location(self, sel_room_id, all_locations):
        
 #       print("sel_room_id = ", sel_room_id)
        
        no_locations=len(all_locations)
        
        poss_goal_locations=[]
        
        for sel_location in np.arange(no_locations):
        
            location_room_id=all_locations[sel_location].room_id
            
            if location_room_id==sel_room_id:
                
                poss_goal_locations=np.hstack([poss_goal_locations, sel_location])
                
#        print("poss_goal_locations = ",poss_goal_locations)
        
        sel_goal=int(np.random.permutation(poss_goal_locations)[0])
        
#        print("sel goal = ", sel_goal)
        
        return(sel_goal)




    ##################################
    
    ##this function sets a new goal destination for the agent
    
    ##Function: Set_New_Goal_Class
    
    ##Inputs: The probability of moving if the agent is a teacher
    
    ##Outputs: The new classroom set as the goal
    
    ##What it does
    
    ##When a new goal is required, the agent is assigned a new classroom to aim to travel to (note: with some probability this will be their current location).

    def Set_New_Goal_Class(self, prob_teacher_moving, canteen_prob, current_lunch_time, prob_follow_group, assigned_teacher_classrooms, all_locations):
        
        old_goal=self.goal_location ##register the old goal, as for the teachers the goal may not change
        
        sel_goal=old_goal
    
        agent_type=self.agent_type ##agent type
        
        if current_lunch_time==0:
        
            poss_goal_locations=self.all_classrooms ##the goals are the classrooms
            
            if agent_type!=2:
            
                sel_room_id=int(np.random.permutation(poss_goal_locations)[0]) ##select one at random
                
                sel_goal=self.Select_Random_Room_Location(sel_room_id, all_locations)
            
            if agent_type==2:
                
                no_classrooms=len(assigned_teacher_classrooms) ##check the number of classrooms
        
                classroom_id=int(np.mod(self.agent_id,no_classrooms)) ##and use the agent ID to work out which class to put the teacher in
                
#                r=np.random.random()
                
 #               if r<prob_teacher_moving: ##...and by chance they don't move
                
                sel_room_id=int(assigned_teacher_classrooms[classroom_id]) ##assign this as the initial location
                
                sel_goal=self.Select_Random_Room_Location(sel_room_id, all_locations)
                
                self.teacher_classroom=sel_goal
                
                #sel_goal=self.teacher_classroom

        if current_lunch_time==1:
            
            if agent_type!=2: ##if the agent is a student....
                
                r=np.random.random()
            
                if r<canteen_prob:
                    
                    sel_room_id=self.all_canteens[0]
                    
                    sel_goal=self.Select_Random_Room_Location(sel_room_id, all_locations)
                    
                else:
                    
                    poss_goal_locations=self.available_locations
                    
                    sel_room_id=int(np.random.permutation(poss_goal_locations)[0]) ##select one at random
                    
                    sel_goal=self.Select_Random_Room_Location(sel_room_id, all_locations)
                    
                    #print("sel_goal = ",sel_goal)
            
 #           if agent_type==2: ##if the agent is a teacher....

  #              sel_goal=self.all_staffrooms[0]

            if agent_type==2: ##if the agent is a teacher....
            
                r=np.random.random()
                
                sel_room_id=self.all_staffrooms[0]
                
                sel_goal=self.Select_Random_Room_Location(sel_room_id, all_locations)
                
                if r>prob_teacher_moving: ##...and by chance they don't move
                
                    sel_goal=old_goal ##set the new goal to the old goal
                    
                if self.canteen_duty==1:
                    
                    sel_room_id=self.all_canteens[0]
                    
                    sel_goal=self.Select_Random_Room_Location(sel_room_id, all_locations)
                    
        if agent_type!=2: ##if the agent is not a teacher....
                
            r=np.random.random()

            if r<prob_follow_group:
                
                own_group=self.movement_group
                
                sel_goal=int(self.group_goals[own_group])
                
      #          print("sel group goal = ", sel_goal)
                
  #      print("sel room id = ",sel_room_id,", Sel goal = ", sel_goal)

        self.goal_location=sel_goal
        
    ##################################
        
    def Set_New_Goal_Toilet(self, all_locations):
        
        old_goal=self.goal_location ##register the old goal, as for the teachers the goal may not change
        
        poss_goal_locations=self.all_toilets ##the goals are the toilets

        sel_room_id=int(np.random.permutation(poss_goal_locations)[0]) ##select one at random
        
        sel_goal=self.Select_Random_Room_Location(sel_room_id, all_locations)
        
        self.goal_location=sel_goal

        
    ##################################
    
    ##this function uses the library nx to find the shortest path to the new goal, which then gives the "ideal path length" if nothing goes wrong
    
    ##Function: Ideal_Goal_Length
    
    ##Inputs: All locations; the network of connected locations, G.
    
    ##Outputs: The shortest path to the agent's goal.
    
    ##What it does
    
    ##This function finds the shortest path the agent should take to their goal.  This therefore provides the time the agent assumes it should take, rather than how long it actually may take due to random events.

    def Ideal_Goal_Length(self,all_locations,G):
    
        current_location=self.current_location

       # print()

        path_to_goal = nx.shortest_path(G, self.current_location, self.goal_location)

        self.ideal_goal_length=len(path_to_goal)

    ##################################
    
    ##this function moves the agent to the next location on the shortest path to their goal
    
    ##Function: Move_Agent
    
    ##Inputs: All locations; the agent's own current location; the network of connected locations, G.
    
    ##Outputs: An updated location (and coordinates) for the agent; an updated journey time
    
    ##What it does
    
    ##This function finds the next step in the shortest route between the agent's current location and their goal, and moves them to this point (and generates some random coordinates within the location for plotting).  It also updates how long the journey has been going on for.

    def Move_Agent(self,all_locations,G):

        current_location=self.current_location

        path_to_goal = nx.shortest_path(G, self.current_location, self.goal_location) ##calculate the shortest path to the goal

        if len(path_to_goal)>1: ##if they have some distance to go
        
            self.journey_time=self.journey_time+1 ##add one to the journey time

            new_location=path_to_goal[1] ##set the new location to the next path

            self.current_location=new_location

            current_location_coords=all_locations[new_location].coords ##and set the new coordinates...

            new_agent_coords_shift=np.random.random(2)-0.5 ##...with a little jiggle for plotting

            new_agent_coords=current_location_coords+new_agent_coords_shift

            self.current_position=new_agent_coords
            
        else:
        
            self.journey_time=0


    ############################################

    ##functions related to relationship quality

    ##################################
    
    ##this function initialises the relationship quality between -1 and 1 between all agents
    
    ##Function: Initialise_RQ
    
    ##Inputs: All agents
    
    ##Outputs: A relationship quality between the agent and others.
    
    ##What it does
    
    ##This function assigns a random number between -1 and 1 as the initial relationship quality between the agents.

    def Initialise_RQ(self, all_agents, emp_networks, initial_agent_types, no_teachers, initial_rq_inputs):
        
        net_size=np.size(emp_networks)

        no_agents=len(all_agents)
        
        no_students=len(np.where(initial_agent_types!=2)[0])

#        print("No students = ",no_students)
        
        sel_agent=self.agent_id
        
        mean_rq=initial_rq_inputs[0]
        
        sd_rq=initial_rq_inputs[1]
        
        random_rq=np.random.normal(loc=mean_rq, scale=sd_rq, size=no_agents)
        
        if net_size>1 and sel_agent>(no_teachers-1):
            
            #print("Sel agent = ", sel_agent)
            
            sel_agent_student_rq=np.array(emp_networks.iloc[:, (sel_agent-no_teachers)])

            #no_network_students=len(all_sel_agent_student_rq)

            #print("No network students = ",no_network_students)

            #sel_agent_student_rq=all_sel_agent_student_rq[0:no_students]

            sel_agent_rq=np.zeros(no_agents)

            sel_agent_rq[initial_agent_types!=2]=random_rq[initial_agent_types!=2]+sel_agent_student_rq

 #       print("sel_agent_rq")

  #      print(sel_agent_rq)

            self.all_rq=sel_agent_rq#np.random.random(no_agents)*2-1
            
        else:
            
            self.all_rq=random_rq

        self.rq=np.sum(self.all_rq)

    ##################################
    
    ##this function calculates the loneliness as a sum of all relationship quality
    
    ##Function: Update_Loneliness_Calc
    
    ##Inputs: The agent's relationship quality with all others
    
    ##Outputs: Loneliness (or really the inverse, which is the total relationship quality).
    
    ##What it does
    
    ##This function sums the agent's relationship quality across all other agent's to get a measure of loneliness.

    def Update_Loneliness_Calc(self):

        self.rq=np.sum(self.all_rq)
        
    ##################################
    
    ##this function updates the relationship quality of an agent based on the previous interactions
    
    ##Function: Update_All_RQ_Based_On_Interaction
    
    ##Inputs: Whether the agent has bullied or been bullied in the previous time step; the change in relationship quality based on whether an agent is bullied or been bullied
    
    ##Outputs: Updated relationship quality between the focal agent and all others
    
    ##What it does
    
    ##This function reduces the relationship quality between two agents if they bully another, or are bullied by another.

    def Update_All_RQ_Based_On_Interaction(self,rq_decrease_through_bullying,rq_increase_through_support):

        agent_type=self.agent_type ##agent type

        if agent_type!=2: ##if the agent is not a teacher....

            if self.bullied_by>-1: ##if the focal agent was bullied

                self.all_rq[self.bullied_by]=self.all_rq[self.bullied_by]-rq_decrease_through_bullying ##decrease the RQ due to the bullying

                self.bullied_by=-1

            if self.bullying>-1: ##if the focal agent bullied

                self.all_rq[self.bullying]=self.all_rq[self.bullying]-rq_decrease_through_bullying ##decrease the RQ due to the bullying

                self.bullying=-1

            if self.supported_by>-1: ##if the focal agent is socially supported

                self.all_rq[self.supported_by]=self.all_rq[self.supported_by]+rq_increase_through_support ##increase the RQ due to the social support

                self.bullied_by=-1

    ############################################

    ##functions related to stress and status

    ##################################
    
    ##this function updates the status based on providing or receiving a negative social interaction
    
    ##Function: Update_Status
    
    ##Inputs: Agent type; the agent's current status; whether a negative interaction was received or provided in the last time step.
    
    ##Outputs: The updated agent status
    
    ##What it does
    
    ##This function increases an agent's status if they have provided a negative interaction, and reduces it if they've been the subject of one.

    def Update_Status(self,status_increase,status_decrease):

        agent_type=self.agent_type

        if agent_type!=2: ##if the agent is a student

            self.status=self.status+status_increase*self.neg_int_prov-status_increase*self.neg_int_rec ##update the status

            if self.status<0: ##make sure the status isn't negative

                self.status=0

            self.neg_int_prov=0

            self.neg_int_rec=0
                
    ##################################
    
    ##this function decays the stress
    
    ##Function: Decay_Stress
     
    ##Inputs: Stress decay rate
    
    ##Outputs: Updated agent stress
    
    ##What it does
    
    ##Decays the stress of students

    def Decay_Stress(self, stress_decay, time):

        agent_type=self.agent_type
        
        base_stress=0
        
        if agent_type!=2: ##if the agent is a student

            current_location=self.current_location
            
            base_stress=self.location_stress[current_location]+self.time_stress[time]

            if base_stress<0:
            
                base_stress=0

        self.stress=self.stress-(self.stress-base_stress)*stress_decay ##update the stress
                
        if self.stress<0:
        
            self.stress=0
                
    ##################################
    
    ##this function adds stress based on the time step
    
    ##Function: Decide_If_Stressed_Due_To_Time
    
    ##Inputs: agent type; agent stress in each time
    
    ##Outputs: Updated agent stress
    
    ##What it does
    
    ##Adds the base time stress to the agent's stress.

    def Decide_If_Stressed_Due_To_Time(self,all_locations,status_threshold,time):

        agent_type=self.agent_type

        if agent_type!=2: ##if the agent is a student

            self.stress=self.stress+self.time_stress[time] ##add the correct stress from the time step
                
    
    
    ##################################
    
    ##this function adds stress based on the location
    
    ##Function: Decide_If_Stressed_Due_To_Location
    
    ##Inputs: agent type; agent stress in each location
    
    ##Outputs: Updated agent stress
    
    ##What it does
    
    ##Adds the base location stress to the agent's stress.

    def Decide_If_Stressed_Due_To_Location(self,all_locations,status_threshold,time):

        agent_type=self.agent_type

        if agent_type!=2: ##if the agent is a student

            current_location=self.current_location

            self.stress=self.stress+self.location_stress[current_location] ##add the correct stress for the location

            
    ##################################
    
    ##this function decides if the agent is stressed due to a delay
    
    ##Function: Decide_If_Stressed_Due_To_Delay
    
    ##Inputs: agent type; agent journey time; agent ideal goal length
    
    ##Outputs: Updated agent stress
    
    ##What it does
    
    ##If the journey between classrooms has taken longer than initially calculated, the agent becomes more stressed.
    
    def Decide_If_Stressed_Due_To_Delay(self,journey_stress):

        agent_type=self.agent_type

        if agent_type!=2: ##if the agent is a student
        
            time_diff=self.journey_time-self.ideal_goal_length ##if the journey time is longer than the time it should have taken when the goal was set
        
            if time_diff>0:

                self.stress=self.stress+journey_stress ##add the journey stress


    ##################################
    
    ##this function makes an agent stressed if there are too many others in the same location
    
    ##Function: Decide_If_Stressed_Due_To_Crowdedness
    
    ##Inputs: agent type; agent location; other agents in the location; other agent types in the location; is the location a classroom?; number of students required in a location to make it stressful; stress due to being in a crowded location.
    
    ##Outputs: Updated agent stress
    
    ##What it does
    
    ##If the agent isn't in a classroom or in a location with a teacher, but the number of other students is larger than the threshold, then increase the agent's stress by the amount given by the parameter.

    def Decide_If_Stressed_Due_To_Crowdedness(self,all_locations,crowded_threshold,crowded_stress):

        agent_type=self.agent_type

        if agent_type!=2: ##if the agent is a student

            current_location=self.current_location

            agents_here=all_locations[current_location].agents_in_location ##check how many agents there are in the current location
            
            no_agents_here=len(agents_here)-1
            
            is_loc_classroom=all_locations[current_location].is_classroom ##check if the location is a classroom (as crowdedness is only a factor outside)

            agent_types_here=all_locations[current_location].no_agent_types_present ##check if there are any teachers in the location
                
            no_teachers=agent_types_here[2]

            if no_agents_here>crowded_threshold and is_loc_classroom==0: ##if the number of others is larger than the threshold, there are no teachers and the location isn't a classroom - add to the stress 

                if no_teachers==0:

                    self.stress=self.stress+crowded_stress
                    
                else:
                    
                    self.stress=self.stress+crowded_stress/2

                    
    ##################################
    
    ##this function updates the stress of an individual due to any positive or negative interactions that have taken place
    
    ##Function: Update_Stress_Due_To_Interactions
    
    ##Inputs: agent type; agent location; agent types in the current location; is the student bullied in the current time step?; is the student supported in the current time step?; increase in stress due to a negative interaction; decrease in stress due to a positive interaction
    
    ##Outputs: Updated agent stress
    
    ##What it does
    
    ##If a student is negatively interacted with (and there are no teachers present), then the student's stress increases by the amount given by the parameter "increase_in_stress_due_to_neg_int", whilst if the student is positively interacted with the student's stress idecreases by the amount given by the parameter "decrease_in_stress_due_to_pos_int".

    def Update_Stress_Due_To_Interactions(self, all_locations, all_agents, increase_in_stress_due_to_neg_int, decrease_in_stress_due_to_pos_int, reduction_due_to_teacher_presence):

        agent_type=self.agent_type

        if agent_type!=2: ##if the agent is a student

            current_location=self.current_location

            agent_types_here=all_locations[current_location].no_agent_types_present ##check if there are any guardians present
                
            am_bullied=self.neg_int_rec ##check if the student has been bullied in the current time step

            no_teachers=agent_types_here[2]

            if am_bullied==1:

                if no_teachers>1: ##if the student is bullied, but there are no teachers increase stress

                    self.stress=self.stress+increase_in_stress_due_to_neg_int*reduction_due_to_teacher_presence
                    
                if no_teachers==0: ##if the student is bullied, but there are no teachers increase stress

                    self.stress=self.stress+increase_in_stress_due_to_neg_int

            am_supported=self.pos_int_rec ##check if the student was supported

            if am_supported==1:

                self.stress=self.stress-decrease_in_stress_due_to_pos_int ##if so, reduce the stress

            if self.stress<0: ##make sure the stress is never negative

                self.stress=0
                
        ##reset the negative interactions from this interaction
                
#        self.neg_int_rec=0
        
 #       self.pos_int_rec=0
            
    ##################################
    
    ##this function updates the stress due to a class interaction (based on e.g., uniform)
    
    ##Function: Update_Stress_Due_To_Class_Interactions
    
    ##Inputs: increase in stress through a class interaction; decrease in stress due to a positive interaction; did the individual receive or instigate a negative interaction based on class?; was the agent supported?
    
    ##Outputs: Updated agent stress
    
    ##What it does
    
    ##This function checks if there was a negative class interaction, and then update the stress accordingly.
            
    def Update_Stress_Due_To_Class_Interactions(self, stress_through_class_interaction, decrease_in_stress_due_to_pos_int):

        am_bullied=self.neg_class_int_rec ##check if there was a negative class interaction

        if am_bullied==1: ##if so, increase stress

            self.stress=self.stress+stress_through_class_interaction

        am_supported=self.pos_int_rec ##if the student is supported, reduce the stress

        if am_supported==1:

            self.stress=self.stress-decrease_in_stress_due_to_pos_int

        class_int_prov=self.neg_class_int_prov ##if the individual is the instigator of the class based interaction, then increase their stress too

        if class_int_prov==1:

            self.stress=self.stress+stress_through_class_interaction
            
        ##set the negative class interaction received or provided back to zero
            
        self.neg_class_int_rec=0
        
        self.neg_class_int_prov=0

        if self.stress<0: ##and then ensure that the stress isn't negative

            self.stress=0
            
    ##################################
    
    ##this function decides if there is a negative interaction between a student and a teacher
    
    ##Function: Decide_Negative_Class_Interaction
    
    ##Inputs: all agent types; current agent location; all agents in the current location; agent classes 
    
    ##Outputs: negative interaction provided; negative interaction received 
    
    ##What it does
    
    ##This function decides if a teacher has a negative interaction based on class.  The focal teacher looks at all students in the same location, and selects one of the students with a lower class to negatively interact with.

    def Decide_Negative_Class_Interaction(self, all_agents, all_locations, standards_interaction_scale):

        agent_type=self.agent_type ##check the agent type

        if agent_type==2: ##if the agent is a teacher

            current_location=self.current_location ##check the current location

            agents_here=all_locations[current_location].agents_in_location ##and check which other agents are here

            teacher_class=self.agent_class ##check the class of the focal teacher

            poss_agents_to_bully=[] ##initialise the vector of the potential agents to interact negatively with

            for sel_agent in agents_here.astype(int): ##cycle through all the agents on the location and check if there are differences in the class between the teacher and others

            #				print("sel_agent = ",sel_agent)

                other_agent_class=all_agents[sel_agent].agent_class

                class_diff=(teacher_class-other_agent_class) ##is the teacher class more strict than the student class?
                
                prob_of_neg_class_int=np.tanh(standards_interaction_scale*class_diff)
                
                r=np.random.random()
                
#                print("Class diff = ",class_diff)
                
                if r<prob_of_neg_class_int: ##and if so add this to the list of possible agents to interact negatively with

                    poss_agents_to_bully.append(sel_agent)
                
                
            ##select an agent to bully

            no_agents_to_bully=len(poss_agents_to_bully)

            if no_agents_to_bully>0: ##if there are any agents to interact negatively with

                sel_agent_to_bully=np.random.permutation(poss_agents_to_bully)[0] ##select one at random

                self.neg_class_int_prov=1 ##and set the bully/bullied variables to 1

                self.bullying=sel_agent_to_bully

                all_agents[sel_agent_to_bully].neg_class_int_rec=1

                all_agents[sel_agent_to_bully].bullied_by_class=self.agent_id

    ##################################
    
    ##this function decides if there is a negative interaction between students based on 1) differences in status and 2) how stressed the focal student is
    
    ##Function: Decide_Negative_Interaction
    
    ##Inputs: agent type; status threshold; stress to bully parameter; status of all agent's in the location; relationship quality between all agent's in the location
    
    ##Outputs: negative interaction provided; negative interaction received
    
    ##What it does
    
    ##This function decides if an agent interacts negatively with another based on the scientific perspective, and if so which other agent.  The agent first compares the status difference (moderated by relationship quality) with all others agents, and if this calculation is larger than the set threshold parameter then the other agent is added to the list of possible agents to interact negatively with.  After this, the agent selects one of the possible agents in the location to interact negatively with.  For the stress level, the agent stress is multiplied by the stress bully scale, and placed into a tanh function to change this into a probability.  The agent then selects another agent in the location to interact with negatively with the probability calculated.

    def Decide_Negative_Interaction(self, all_agents, all_locations, status_threshold, stress_bully_scale):

        agent_type=self.agent_type ##check the agent type

        current_location=self.current_location ##check the current location

        if agent_type!=2: ##if the agent is a student

            own_status=self.status ##check the status of the focal individual

            current_rqs=self.all_rq ##and the relationship quality with all others

            #			print("current_rqs")

            #			print(current_rqs)

            agent_types_here=all_locations[current_location].no_agent_types_present ##and check which other agent types are here

            agent_status_here=all_locations[current_location].agent_status ##and what their status is

            agents_here=all_locations[current_location].agents_in_location ##and check which other agents are here

            poss_agents_to_bully=[] ##initialise a vector of those to bully

            for sel_agent in agents_here.astype(int): ##and cycle through all the agents present to check which satisfy the conditions for a negative interaction (given as a combination of the status difference and the relationship quality between the two

            #				print("sel_agent = ",sel_agent)

                other_agent_status=all_agents[sel_agent].status ##the other agents status

                rq_with_other_agent=current_rqs[sel_agent] ##and relationship quality

            #				print("rq = ",rq_with_other_agent)

                status_diff=(other_agent_status-own_status)*rq_with_other_agent ##choose to interact negatively if have a negative rq and their status is small enough, and then compare this multiplication to the status threshold

#                print("status_diff = ",status_diff)

                if status_diff>status_threshold: ##if this quantity is above a threshold

                    poss_agents_to_bully.append(sel_agent) ##add to the possible agents to bully
                
                
            ##select an agent to bully

            no_agents_to_bully=len(poss_agents_to_bully) 

            if no_agents_to_bully>0: ##if there are agents to bully
                
                #print("Bully")

                sel_agent_to_bully=np.random.permutation(poss_agents_to_bully)[0] ##select one at random

                self.neg_int_prov=1 ##and set the bully/bullied variables to 1

                self.bullying=sel_agent_to_bully

                all_agents[sel_agent_to_bully].neg_int_rec=1

                all_agents[sel_agent_to_bully].bullied_by=self.agent_id
                
        ###################
        
        ##also interact negatively if the agent is stressed
        
        agent_stress=self.stress ##the focal agent stress

        probability_of_negative_interaction=np.tanh(stress_bully_scale*agent_stress) ##calculate the probability of a negative interaction based on the agent's stress

#        print("agent_stress = ",agent_stress,", prob of int = ",probability_of_negative_interaction)

        r=np.random.random() ##generate a random number...
        
        if r<probability_of_negative_interaction: ##...and if this is smaller that the probability of interaction, select another agent to interact negatively with
        
            ##select another agent to interact negatively with
            
            agents_here=all_locations[current_location].agents_in_location ##check which agents are in the location
            
            this_agent_id=self.agent_id
            
         #   print("Agents here")
            
          #  print(agents_here)
            
            other_agents_here=agents_here[np.where(agents_here!=this_agent_id)[0]] ##remove the focal agent from the possibles list
            
           # print("Other agents here")
            
            #print(other_agents_here)
            
            if len(other_agents_here)>1: ##if there are other agents, pick one at random to negatively interact with
            
                sel_agent_to_bully=int(np.random.permutation(other_agents_here)[0])

             #   print("sel_agent_to_bully = ",sel_agent_to_bully)

                self.neg_int_prov=1 ##and set the bully/bullied variables to 1

                self.bullying=sel_agent_to_bully

                all_agents[sel_agent_to_bully].neg_int_rec=1

                all_agents[sel_agent_to_bully].bullied_by=self.agent_id
            
            
            
    ##################################
    
    ##this function decides if there should be a positive interaction between students
    
    ##Function: Decide_Positive_Interaction
    
    ##Inputs: agent type; current agent location; relationship quality between all agents in the location; whether each agent in the location has received a negative interaction
    
    ##Outputs: positive support provided; positive support received
    
    ##What it does
    
    ##This function decides if social support is provided between individuals.  The agent identifies those friends (those other agents in the location with a positive relationship quality) that are being bullied, and selects one at random to socially support.
            
    def Decide_Positive_Interaction(self,all_agents,all_locations):

        ##find the friends in the location

        agent_type=self.agent_type ##check the agent type

        if agent_type!=2: ##if the agent is a student

            current_location=self.current_location ##check the current location,

            current_rqs=self.all_rq ##check the focal agents relationship quality with all other agents

            #			print("current_rqs")

            #			print(current_rqs)

            agents_here=(all_locations[current_location].agents_in_location).astype(int) ##and check which other agents are here

            #			print("agents_here")

            #			print(agents_here)

            location_rqs=current_rqs[agents_here] ##and then select the relationship quality of all others present

            #			print("location_rqs")

            #			print(location_rqs)

            friends_loc=np.where(location_rqs>0)[0] ##find which part of the previous vector includes those that have a positive relationship quality...

            location_friends=agents_here[friends_loc] ##...and using this identify which individuals in the same location are friends

            #			print("location_friends")

            #			print(location_friends)

            no_location_friends=len(location_friends)

            bullied_friends=np.zeros(no_location_friends) ##initialise a vector to check which friends are being bullied, and therefore require support

            for sel_friend_id in np.arange(no_location_friends): ##for each friend in the current location...

                sel_friend=location_friends[sel_friend_id]

                bullied_friends[sel_friend_id]=all_agents[sel_friend].neg_int_rec ##check if that friend is bullied

                #			print("bullied_friends")

                #			print(bullied_friends)

            no_bullied_friends=np.sum(bullied_friends) ##check how many bullied friends there are in the location 

            sel_supported_friend=-1 ##initialise which friend to support

            if no_location_friends>0: ##if there are friends present...

                if no_bullied_friends>0: ##...and if at least one of them is being bullied

                    poss_supported_friends_ids=np.where(bullied_friends==1)[0]
                    
                    poss_supported_friends=location_friends[poss_supported_friends_ids]
                    
                    sel_supported_friend=np.random.permutation(poss_supported_friends)[0] ##select one of these friends at random to support
                    
                else:

                    sel_supported_friend=np.random.permutation(location_friends)[0] ##otherwise, just choose a non-bullied friend to have a positive interaction with

                self.pos_int_prov=1 ##set the agent indicators to show they have provided and received a positive interaction

                self.supporting=sel_supported_friend

                all_agents[sel_supported_friend].pos_int_rec=1

                all_agents[sel_supported_friend].supported_by=self.agent_id

                #			print("sel_supported_friend")

                #			print(sel_supported_friend)

    ##################################
    
    def Calc_Network_Degree(self):
    
        ind_rq=self.all_rq
        
        pos_deg=0
        
        neg_deg=0
        
        for partner_rq in ind_rq:
            
            if partner_rq>5:
                
                pos_deg=pos_deg+1
                
            if partner_rq<-80:
                
                neg_deg=neg_deg+1

        self.pos_deg=pos_deg
        
        self.neg_deg=neg_deg
        
    ##################################
    
    ##this function calculates how lonely individuals are in each situation
    
    def Calc_Situational_Loneliness(self, all_locations):
        
        ##find the friends in the location

        agent_type=self.agent_type ##check the agent type

        if agent_type!=2: ##if the agent is a student

            current_location=self.current_location ##check the current location,

            current_rqs=self.all_rq ##check the focal agents relationship quality with all other agents

            #			print("current_rqs")

            #			print(current_rqs)

            agents_here=(all_locations[current_location].agents_in_location).astype(int) ##and check which other agents are here

            #			print("agents_here")

            #			print(agents_here)

            location_rqs=current_rqs[agents_here] ##and then select the relationship quality of all others present
            
            self.location_loneliness=np.sum(location_rqs) ##add up the location RQs for the loneliness of the student
            
            


    ##################################
    
    ##this function records all of the necessary information in this time step for this agent
 
    def Record_Agent_Info(self,time):

        self.agent_info[time,[0,1]]=self.current_position

        self.agent_info[time,2]=self.agent_type
        
        self.agent_info[time,3]=self.stress
        
        self.agent_info[time,4]=self.location_loneliness#self.rq
        
        self.agent_info[time,5]=self.status
        
        self.agent_info[time,6]=self.pos_deg
        
        self.agent_info[time,7]=self.neg_deg

        self.agent_info[time,8]=self.current_location
        
        








