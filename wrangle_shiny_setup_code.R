##code to generate the empirical data used in the SOCITS ABM model calibration, and also the input text for the 
##model function and the Shiny application

###############

# Clear the environment
rm(list=ls())

library(tidyverse)
library(data.table)

working_dir <- "T:/projects/SOCITS S00606/Data/AnonymisedData/Surveys"

# Set the workspace path to the directory containing this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#############################################################################################

##load in the model input file

model_input_pars<-data.table(read.csv("model_setup_pars.csv"))

##now generate a text file for the correct entry and order

model_input_pars[,par_count:=0:(.N-1)]

model_input_pars[,model_input_text:=paste0("model_inputs[",par_count,"]=",Set)]

model_input_pars[shiny_input=="Y", model_input_text:=paste0("model_inputs[",par_count,"]=input.", Parameter_name,"()")]

model_input_pars[shiny_input=="Y", shiny_input_text:=paste0("ui.input_slider(\"", Parameter_name,"\", \"",Text,"\", ", Min,", ", Max,", ", Set,"),")]

model_input_pars[shiny_input=="Y" & Type=="int", shiny_input_text:=paste0("ui.input_slider(\"", Parameter_name,"\", \"",Text,"\", ", Min,", ", Max,", ", Set,", step = 1),")]


##and save this text to a file

write.csv(model_input_pars[, c("model_input_text", "shiny_input_text")], "shiny_input_text.csv", row.names = F)









