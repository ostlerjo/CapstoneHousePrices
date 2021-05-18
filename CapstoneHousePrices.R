#Capstone House Prices Project
#Jonathan Ostler
#May 2021

#Testing github to see if anything changes
#Need to add in more libraries

###Install and load libraries
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Boruta)) install.packages("Boruta", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

library(caret)
library(data.table)
library(Boruta)
library(plyr)
library(dplyr)
library(pROC)
library(tidyverse)

###Download data
dl <- read_csv("data/train.csv")



###Clean data

#NA's , 0's and X1st etc

#Huge amount of factoring and manipulating - is this necessary????? Look at comments section


###Split Data
# Ensure naming is consistent everywhere - train vs. training vs. final vs. validation etc - also boruta uses a different set because training gets split/overwritten
#90 / 10

#And then 80 /20


###Explore Data

#Exploring dataset could be diffcult when the quantity of variables is quite huge. Therefore, I mainly focused on the exploration of numeric
#variables in this report. The descriptive analysis of dummy variables are mostly finished by drawing box plots. Some dummy variables, like 'Street',
#are appeared to be ineffective due to the extreme box plot. The numeric variables are sorted out before turning dummy variables into numeric form.



##Whole Boruta thing
#Check which data source is used


##3 correlation plots from Fun With + 1 from XGBoost

##Simple scatterplot matrix (pairs) + 1 from XGBoost (pairs)

##3 scatterplot charts - not sure if they really show anything!!

## Sales Price vs. Year Built => newer houses worth more

###Create Models

##Average Price
#RMSE

##Linear Model All attributes
#RMSE

##Linear Model Correlation attributes (17)
#Plot the summary => 4 charts in a grid
#RMSE

##Linear Model Confirmed Attributes (49)
#RMSE

##LASSO Regression
#Plot LASSO
#RMSE

##Random Forest
#RMSE

##XGBoost
#RMSE











###Validate data



###Save Environment Variables
#Look at one of the MovieLens scripts


