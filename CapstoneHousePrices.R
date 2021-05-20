#Capstone House Prices Project
#Jonathan Ostler
#May 2021

###Install and load libraries
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Boruta)) install.packages("Boruta", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(FeatureHashing)) install.packages("FeatureHashing", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(dummies)) install.packages("dummies", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(car)) install.packages("car", repos = "http://cran.us.r-project.org")
if(!require(lars)) install.packages("lars", repos = "http://cran.us.r-project.org")

library(caret)
library(data.table)
library(Boruta)
library(plyr)
library(dplyr)
library(pROC)
library(tidyverse)
library(FeatureHashing)
library(Matrix)
library(xgboost)
require(randomForest)
require(ggplot2)
library(stringr)
library(dummies)
library(Metrics)
library(kernlab)
library(corrplot)
library(car)
library(lars)

###Download data
dl <- read_csv("data/train.csv")   #Read in as factors???
train <- dl

###Initial Data Clean

#Rename a couple of columns
names(train)[names(train) == '1stFlrSF'] <- 'X1stFlrSF'
names(train)[names(train) == '2ndFlrSF'] <- 'X2ndFlrSF'
names(train)[names(train) == '3SsnPorch'] <- 'X3SsnPorch'

#Update "wrong" NAs to real values
Num_NA<-sapply(train,function(y)length(which(is.na(y)==T)))
NA_Count<- data.frame(Item=colnames(train),Count=Num_NA)
NA_Count

train$Alley[is.na(train$Alley)] <- "NoAlleyAccess"
train$PoolQC[is.na(train$PoolQC)] <- "NoPool"
train$Fence[is.na(train$Fence)] <- "NoFence"
train$MiscFeature[is.na(train$MiscFeature)] <- "None"
train$BsmtQual[is.na(train$BsmtQual)] <- "NoBasement"
train$BsmtCond[is.na(train$BsmtCond)] <- "NoBasement"
train$BsmtExposure[is.na(train$BsmtExposure)] <- "NoBasement"
train$BsmtFinType1[is.na(train$BsmtFinType1)] <- "NoBasement"
train$BsmtFinType2[is.na(train$BsmtFinType2)] <- "NoBasement"
train$FireplaceQu[is.na(train$FireplaceQu)] <- "NoFireplace"
train$GarageType[is.na(train$GarageType)] <- "NoGarage"
train$GarageFinish[is.na(train$GarageFinish)] <- "NoGarage"
train$GarageQual[is.na(train$GarageQual)] <- "NoGarage"
train$GarageCond[is.na(train$GarageCond)] <- "NoGarage"
train$GarageYrBlt[is.na(train$GarageYrBlt)] <- "NoGarage"
train$MasVnrType[is.na(train$MasVnrType)] <- "None"
train$Electrical[is.na(train$Electrical)] <- "NotApplicable"

#Update two sets of NAs to 0
train$MasVnrArea[is.na(train$MasVnrArea)] <- 0
train$LotFrontage[is.na(train$LotFrontage)] <- 0

#Check if process worked
Num_NA<-sapply(train,function(y)length(which(is.na(y)==T)))
NA_Count<- data.frame(Item=colnames(train),Count=Num_NA)
NA_Count

###Split/Partition Data
#The "test" data provided does not have a column called "SalePrice" since it's used for a competition and is therefore secret
#For the purposes of this project, I will use 10% of the "train" data as the final validation data set
dim(train)
outcome1 <- train$SalePrice
length(outcome1)

set.seed(1, sample.kind = "Rounding")
partition1 <- createDataPartition(y=outcome1, p=0.1, list=F)

jFinal <- train[partition1,]
dim(jFinal)

jModelling <- train[-partition1,]
dim(jModelling)

#The modelling set of data is further split 60 / 40 in favour of the training set
outcome2 <- jModelling$SalePrice
length(outcome2)

set.seed(2, sample.kind = "Rounding")
partition2 <- createDataPartition(y=outcome2, p=0.4, list=F)

jTesting <- jModelling[partition2,]
dim(jTesting)

jTraining <- jModelling[-partition2,]
dim(jTraining)

###Data Prep
# Original Numeric Variables - Used in correlation matrices later on
Num <- sapply(jTraining,is.numeric)
Num <- jTraining[,Num]
numberCols <- names(Num)
numberCols <- numberCols[-1] #remove Id column



#################################
#################################
#Huge amount of factoring and manipulating - is this necessary????? Look at comments section
# If read in as factors, no need to convert to numbers????
# train$heatqual <- as.integer(train$heatqual)
#Reduced the changes to 5 + 1 variables - see a bit later
#################################
#################################

###Explore Data

##Boruta Feature Importance
#Exploring dataset could be difficult when the number of variables is as large as it is (Id + 79 + SalePrice).
#Start with a Boruta Feature Importance analysis to show which items are deemed important

#Determine data type for each candidate explanatory attributes
ID.VAR <- "Id"
TARGET.VAR <- "SalePrice"

#Extract only candidate feature names i.e exclude the ID column and the SalesPrice column
candidate.features <- setdiff(names(jTraining),c(ID.VAR,TARGET.VAR))
data.type <- sapply(candidate.features,function(x){class(jTraining[[x]])})
table(data.type)

print(data.type)

#Determine data types
explanatory.attributes <- setdiff(names(jTraining),c(ID.VAR,TARGET.VAR))
data.classes <- sapply(explanatory.attributes,function(x){class(jTraining[[x]])})

#Categorise data types in the data set
unique.classes <- unique(data.classes)

attr.data.types <- lapply(unique.classes,function(x){names(data.classes[data.classes==x])})
names(attr.data.types) <- unique.classes

###Prepare data set for Boruta analysis
#Pull out the response variable
response <- jTraining$SalePrice

#Remove identifier and response variables
dependents <- jTraining[candidate.features]

#Run Boruta Analysis
set.seed(13, sample.kind = "Rounding")
bor.results <- Boruta(dependents , response,
                      maxRuns=101,
                      doTrace=0)

#Boruta results
print(bor.results)

#These attributes were deemed as relevant to predicting house sale price.
getSelectedAttributes(bor.results)

#The following plot shows the relative importance of each candidate explanatory attribute.
#The x-axis represents each of candidate explanatory variables.
#Green color indicates the attributes that are relevant to prediction.
#Red indicates attributes that are not relevant.
#Yellow color indicates attributes that may or may not be relevant to predicting the response variable.
plot(bor.results)

#Detailed results for each candidate explanatory attributes.
impTable <- arrange(cbind(attr=rownames(attStats(bor.results)), attStats(bor.results)),desc(medianImp))
impTable

#Extract the three types of importance
confirmed_tib <- getSelectedAttributes(bor.results, withTentative = FALSE)
tent_n_confirmed_tib <- getSelectedAttributes(bor.results, withTentative = TRUE)

confirmed.df <- as.data.frame(confirmed_tib)
tent_n_confirmed.df <- as.data.frame(tent_n_confirmed_tib)
all.df <- as.data.frame(candidate.features)

tentative <- tent_n_confirmed.df %>% anti_join(confirmed.df, by=c("tent_n_confirmed_tib"="confirmed_tib"))
rejected <- all.df %>% anti_join(tent_n_confirmed.df, by=c("candidate.features" = "tent_n_confirmed_tib"))
confirmed <- confirmed.df

#Lot of dependents therefore, I mainly focused on the exploration of numeric
#variables in this report. The descriptive analysis of dummy variables are mostly finished by drawing box plots. Some dummy variables, like 'Street',
#are appeared to be ineffective due to the extreme box plot. The numeric variables are sorted out before turning dummy variables into numeric form.

#All original numeric values
correlations <- cor(jTraining[, numberCols],use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

#13 of the 35 (36) variables are of interest (high correlations with SalePrice)
NamesInterest1 <- names(which(correlations["SalePrice", ] > 0.35))
NamesInterest1 <- NamesInterest1[!NamesInterest1 %in% "SalePrice"] #Remove SalePrice

#Split the numeric values coming from the Boruta analysis
numeric_confirmed <- intersect(confirmed_tib, numberCols)
numeric_confirmed_SalePrice <- c(numeric_confirmed, "SalePrice")
correlations <- cor(jTraining[, numeric_confirmed_SalePrice],use="everything") #Columns that were originally numeric
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

#12 of the 23 (24) variables are of interest (high correlations with SalePrice)
NamesInterest2 <- names(which(correlations["SalePrice", ] > 0.35))
NamesInterest2 <- NamesInterest2[!NamesInterest2 %in% "SalePrice"] #Remove SalePrice

#From the Boruta analysis, of the top 20 variables only 5 are character variables
#Look at the important character variables
charCols <- attr.data.types$character
char_confirmed <- intersect(confirmed_tib, charCols)


#Let's convert those to factors and then run one more set of correlations -> ALSO DOING THIS IN jTESTING and jFINAL data sets
#GarageType
price <- jTraining %>%
  group_by(GarageType) %>%
  summarize(avg=mean(SalePrice, na.rm=T)) %>%
  arrange(desc(avg))

jTraining$jGarageType[jTraining$GarageType %in% c("BuiltIn", "Attchd")] <- 3
jTraining$jGarageType[jTraining$GarageType %in% c("Basment", "Detchd", "2Types")] <- 2
jTraining$jGarageType[jTraining$GarageType %in% c("CarPort", "NoGarage")] <- 1

jTraining$GarageType <- NULL

jTesting$jGarageType[jTesting$GarageType %in% c("BuiltIn", "Attchd")] <- 3
jTesting$jGarageType[jTesting$GarageType %in% c("Basment", "Detchd", "2Types")] <- 2
jTesting$jGarageType[jTesting$GarageType %in% c("CarPort", "NoGarage")] <- 1

jTesting$GarageType <- NULL

jFinal$jGarageType[jFinal$GarageType %in% c("BuiltIn", "Attchd")] <- 3
jFinal$jGarageType[jFinal$GarageType %in% c("Basment", "Detchd", "2Types")] <- 2
jFinal$jGarageType[jFinal$GarageType %in% c("CarPort", "NoGarage")] <- 1

jFinal$GarageType <- NULL

#KitchenQual
price <- jTraining %>%
  group_by(KitchenQual) %>%
  summarize(avg=mean(SalePrice, na.rm=T)) %>%
  arrange(desc(avg))

jTraining$jKitchenQual[jTraining$KitchenQual == "Ex"] <- 4
jTraining$jKitchenQual[jTraining$KitchenQual == "Gd"] <- 3
jTraining$jKitchenQual[jTraining$KitchenQual == "TA"] <- 2
jTraining$jKitchenQual[jTraining$KitchenQual == "Fa"] <- 1
jTraining$jKitchenQual[jTraining$KitchenQual == "Po"] <- 0

jTraining$KitchenQual <- NULL

jTesting$jKitchenQual[jTesting$KitchenQual == "Ex"] <- 4
jTesting$jKitchenQual[jTesting$KitchenQual == "Gd"] <- 3
jTesting$jKitchenQual[jTesting$KitchenQual == "TA"] <- 2
jTesting$jKitchenQual[jTesting$KitchenQual == "Fa"] <- 1
jTesting$jKitchenQual[jTesting$KitchenQual == "Po"] <- 0

jTesting$KitchenQual <- NULL

jFinal$jKitchenQual[jFinal$KitchenQual == "Ex"] <- 4
jFinal$jKitchenQual[jFinal$KitchenQual == "Gd"] <- 3
jFinal$jKitchenQual[jFinal$KitchenQual == "TA"] <- 2
jFinal$jKitchenQual[jFinal$KitchenQual == "Fa"] <- 1
jFinal$jKitchenQual[jFinal$KitchenQual == "Po"] <- 0

jFinal$KitchenQual <- NULL

#ExterQual
price <- jTraining %>%
  group_by(ExterQual) %>%
  summarize(avg=mean(SalePrice, na.rm=T)) %>%
  arrange(desc(avg))

jTraining$jExterQual[jTraining$ExterQual == "Ex"] <- 4
jTraining$jExterQual[jTraining$ExterQual == "Gd"] <- 3
jTraining$jExterQual[jTraining$ExterQual == "TA"] <- 2
jTraining$jExterQual[jTraining$ExterQual == "Fa"] <- 1
jTraining$jExterQual[jTraining$ExterQual == "Po"] <- 0

jTraining$ExterQual <- NULL

jTesting$jExterQual[jTesting$ExterQual == "Ex"] <- 4
jTesting$jExterQual[jTesting$ExterQual == "Gd"] <- 3
jTesting$jExterQual[jTesting$ExterQual == "TA"] <- 2
jTesting$jExterQual[jTesting$ExterQual == "Fa"] <- 1
jTesting$jExterQual[jTesting$ExterQual == "Po"] <- 0

jTesting$ExterQual <- NULL

jFinal$jExterQual[jFinal$ExterQual == "Ex"] <- 4
jFinal$jExterQual[jFinal$ExterQual == "Gd"] <- 3
jFinal$jExterQual[jFinal$ExterQual == "TA"] <- 2
jFinal$jExterQual[jFinal$ExterQual == "Fa"] <- 1
jFinal$jExterQual[jFinal$ExterQual == "Po"] <- 0

jFinal$ExterQual <- NULL

#MSZoning
price <- jTraining %>%
        group_by(MSZoning) %>%
        summarize(avg=mean(SalePrice, na.rm=T)) %>%
        arrange(desc(avg))

jTraining$jMSZoning[jTraining$MSZoning == "FV"] <- 4
jTraining$jMSZoning[jTraining$MSZoning == "RL"] <- 3
jTraining$jMSZoning[jTraining$MSZoning %in% c("RH","RM")] <- 2
jTraining$jMSZoning[jTraining$MSZoning == "C (all)"] <- 1

jTraining$MSZoning <- NULL

jTesting$jMSZoning[jTesting$MSZoning == "FV"] <- 4
jTesting$jMSZoning[jTesting$MSZoning == "RL"] <- 3
jTesting$jMSZoning[jTesting$MSZoning %in% c("RH","RM")] <- 2
jTesting$jMSZoning[jTesting$MSZoning == "C (all)"] <- 1

jTesting$MSZoning <- NULL

jFinal$jMSZoning[jFinal$MSZoning == "FV"] <- 4
jFinal$jMSZoning[jFinal$MSZoning == "RL"] <- 3
jFinal$jMSZoning[jFinal$MSZoning %in% c("RH","RM")] <- 2
jFinal$jMSZoning[jFinal$MSZoning == "C (all)"] <- 1

jFinal$MSZoning <- NULL

#Neighborhood
price <- jTraining %>%
  group_by(Neighborhood) %>%
  summarize(avg=mean(SalePrice, na.rm=T)) %>%
  arrange(desc(avg))

price_hi <- filter(price, avg >= 200000)
price_med <- filter(price, avg >= 140000 & avg < 200000)
price_lo <- filter(price, avg < 140000)

jTraining$jNeighborhood[jTraining$Neighborhood %in% price_hi$Neighborhood] <- 3
jTraining$jNeighborhood[jTraining$Neighborhood %in% price_med$Neighborhood] <- 2
jTraining$jNeighborhood[jTraining$Neighborhood %in% price_lo$Neighborhood] <- 1

jTraining$Neighborhood <- NULL

jTesting$jNeighborhood[jTesting$Neighborhood %in% price_hi$Neighborhood] <- 3
jTesting$jNeighborhood[jTesting$Neighborhood %in% price_med$Neighborhood] <- 2
jTesting$jNeighborhood[jTesting$Neighborhood %in% price_lo$Neighborhood] <- 1

jTesting$Neighborhood <- NULL

jFinal$jNeighborhood[jFinal$Neighborhood %in% price_hi$Neighborhood] <- 3
jFinal$jNeighborhood[jFinal$Neighborhood %in% price_med$Neighborhood] <- 2
jFinal$jNeighborhood[jFinal$Neighborhood %in% price_lo$Neighborhood] <- 1

jFinal$Neighborhood <- NULL

#Utilities - Otherwise lm fails because all values = "AllPub"
jTraining$jUtilities[jTraining$Utilities == "AllPub"] <- 1
jTraining$jUtilities[jTraining$Utilities != "AllPub"] <- 0

jTraining$Utilities <- NULL

jTesting$jUtilities[jTesting$Utilities == "AllPub"] <- 1
jTesting$jUtilities[jTesting$Utilities != "AllPub"] <- 0

jTesting$Utilities <- NULL

jFinal$jUtilities[jFinal$Utilities == "AllPub"] <- 1
jFinal$jUtilities[jFinal$Utilities != "AllPub"] <- 0

jFinal$Utilities <- NULL

names(jTraining)
names(jTesting)
names(jFinal)

#Look at the correlations for the character variables
charCols <- c("jGarageType", "jKitchenQual", "jExterQual", "jMSZoning", "jNeighborhood")
charColsSP <- c(charCols, "SalePrice")
correlations <- cor(jTraining[, charColsSP],use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

#Fun with Real Estate commentary
#Another thing I want to do is build some interactions that may be worth looking at. 
#For example, if the house has a pool, is it more important that it has a big deck, or something like that? 
#I used correlation visuals like this to do it- you can choose what you'd want to put in and how many variations
#you want to make.

#XGB commentary on cor
#'OverallQual','TotalBsmtSF','GarageCars' and 'GarageArea' have relative strong correlation with each other.
#Therefore, as an example, we plot the correlation among those four variables and SalePrice.
#Jon - ADD OTHERS IN - 1stFlrSF and GrLivArea

#Come up with an overall list of variable we will include...
#Confirmed, numeric plus 5 character ones




##Simple scatterplot matrix (pairs) + 1 from XGBoost (pairs)
###### What is a scatterplot matrix???? What is "pairs"??? Maybe to subsequently exclude variables that are too closely correlated?????

#The dependent variable (SalePrice) looks having decent linearity when plotting with other variables. However, it is also obvious that some independent variables 
#also have linear relationships with each other.
#The problem of multicollinearity is obvious and should be treated when the quantity of variables in regression formula is huge.

#I picked a few of the variables that had a lot of correlation strengths. Basements have been getting bigger over time, apparently.
#As have the sizes of the living areas. Good to know!

pairs(~YearBuilt+OverallQual+TotalBsmtSF+GrLivArea,data=jTraining, main="Simple Scatterplot Matrix")      ### FunWithRE code
pairs(~SalePrice+OverallQual+TotalBsmtSF+GarageCars+GarageArea,data=jTraining, main="Scatterplot Matrix")  ### XGBoost code

#Too many!!!
pairs(~SalePrice+GrLivArea+OverallQual+TotalBsmtSF+X1stFlrSF+GarageArea+YearBuilt,data=jTraining, main="Scatterplot Matrix")  ### XGBoost code


##3 scatterplot charts - not sure if they really show anything!!
####### What are scatterplots??? What are they actually showing?
#I'm also interested in the relationship between sale price and some numeric variables, but these can be tougher to visualize.
scatterplot(SalePrice ~ YearBuilt, data=jTraining,  xlab="Year Built", ylab="Sale Price", grid=FALSE)
scatterplot(SalePrice ~ YrSold, data=jTraining,  xlab="Year Sold", ylab="Sale Price", grid=FALSE)
scatterplot(SalePrice ~ X1stFlrSF, data=jTraining,  xlab="Square Footage Floor 1", ylab="Sale Price", grid=FALSE)

## Sales Price vs. Year Built => newer houses worth more
#The final descriptive analysis I put here would be the relationship between the variable YearBuilt and SalePrice.
#Merge below with first scatter plot
ggplot(train,aes(x= YearBuilt,y=SalePrice))+
  geom_point()+
  geom_smooth()
#It is not difficult to find that the price of house increases generally with the year built, the trend is obvious. 
#Prices are higher for new houses, that makes sense. Also, we can see that sale prices dropped when we would expect.
#We also have some strange outliers on first floor square footage - probably bad data but it's not going to have a huge influence.



#Based on the above data exploration, it makes sense to use a reduced list of XYZ variables going forward in the creation of the models


###Create Models

##Average Price
mu_price <- mean(jTraining$SalePrice)
mu_price

#RMSE
rmse_mu <- rmse(log(jTesting$SalePrice), log(mu_price))
rmse_mu

#Keep a record/tally of the results as the different models are built up
rmse_results <- tibble(Model = "Average Sale Price", "Log RMSE" = rmse_mu)

#Make table pretty
rmse_results %>% knitr::kable()

##Linear Model - Important Variables
#Finally, we have our data and can build some models. Since our outcome is a continuous numeric variable, 
#we want a linear model, not a GLM.
model_lm <- lm(SalePrice ~ ., data=jTraining[, c(NamesInterest1, charCols , "SalePrice")])
summary(model_lm)

prediction_lm <- predict(model_lm, jTesting, type="response")

#The R Square is not bad, and all variables pass the Hypothesis Test. The diagonsis of residuals is also not bad. The diagnosis can be viewed below.
layout(matrix(c(1,2,3,4), 2, 2, byrow = TRUE))
plot(model_lm)
par(mfrow=c(1,1))

#RMSE
rmse_lm <- rmse(log(jTesting$SalePrice), log(prediction_lm))

#Add to tally
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model = "Linear Model",
                                     "Log RMSE" = rmse_lm ))

#Make table pretty
rmse_results %>% knitr::kable()

###LASSO Regression - Numeric Columns
#For the avoidance of multicollinearity, implementing LASSO regression is not a bad idea. Transferring the variables into the form of matrix, we can automate
#the selection of variables by implementing 'lars' method in Lars package.

Independent_variable <- jTraining[, numberCols]
Independent_variable$SalePrice <- NULL   #Remove SalePrice
Independent_variable <- as.matrix(Independent_variable)

Dependent_Variable <- jTraining[, "SalePrice"]
Dependent_Variable<- as.matrix(Dependent_Variable)

model_lars <- lars(Independent_variable , Dependent_Variable,type = 'lasso')
plot(model_lars)

#The plot is messy as the quantity of variables is intimidating. Despite that, we can still use R to find out the model with least multicollinearity. The selection 
#procedure is based on the value of Marrow's cp, an important indicator of multicollinearity. The prediction can be done by the script-chosen best step and RMSE can be used
#to assess the model.
best_step <- model_lars$df[which.min(model_lars$Cp)]

Testing_variable <- jTesting[, numberCols]
Testing_variable$SalePrice <- NULL   #Remove SalePrice
Testing_variable <- as.matrix(Testing_variable)

prediction_lars <- predict.lars(model_lars , newx = Testing_variable, s=best_step, type= "fit")

#RMSE
rmse_lars <- rmse(log(jTesting$SalePrice),log(prediction_lars$fit))

#Add to tally
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model = "Lasso Model - Numeric",
                                     "Log RMSE" = rmse_lars ))

#Make table pretty
rmse_results %>% knitr::kable()

###LASSO Regression - 2 - Important Variables
Independent_variable_2 <- jTraining[, c(NamesInterest1, charCols)]
#Independent_variable_2$SalePrice <- NULL   #Remove SalePrice
Independent_variable_2 <- as.matrix(Independent_variable_2)

Dependent_Variable_2 <- jTraining[, "SalePrice"]
Dependent_Variable_2 <- as.matrix(Dependent_Variable_2)

model_lars_2 <- lars(Independent_variable_2 , Dependent_Variable_2 , type = 'lasso')
plot(model_lars_2)

#The plot is messy as the quantity of variables is intimidating. Despite that, we can still use R to find out the model with least multicollinearity. The selection 
#procedure is based on the value of Marrow's cp, an important indicator of multicollinearity. The prediction can be done by the script-chosen best step and RMSE can be used
#to assess the model.
best_step_2 <- model_lars_2$df[which.min(model_lars_2$Cp)]

Testing_variable_2 <- jTesting[, c(NamesInterest1, charCols)]
#Testing_variable_2$SalePrice <- NULL   #Remove SalePrice
Testing_variable_2 <- as.matrix(Testing_variable_2)

prediction_lars_2 <- predict.lars(model_lars_2 , newx = Testing_variable_2, s=best_step_2, type= "fit")

#RMSE
rmse_lars_2 <- rmse(log(jTesting$SalePrice),log(prediction_lars_2$fit))

#Add to tally
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model = "Lasso Model - Important",
                                     "Log RMSE" = rmse_lars_2 ))

#Make table pretty
rmse_results %>% knitr::kable()

##Random Forest
#Let's try training the model with an RF.
#Let's use all the variables and see how things look, since randomforest does its own feature selection.
model_rf <- randomForest(SalePrice ~ ., data=jTraining)

# Predict using the test set
prediction_rf <- predict(model_rf, jTesting)

#RMSE
rmse_rf <- rmse(log(jTesting$SalePrice), log(prediction_rf))

#Add to tally
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model = "Random Forest",
                                     "Log RMSE" = rmse_rf ))

#Make table pretty
rmse_results %>% knitr::kable()

##XGBoost

#Assemble and format the data
XGBtraining <- jTraining
XGBtesting <- jTesting

XGBtraining$log_SalePrice <- log(XGBtraining$SalePrice)
XGBtesting$log_SalePrice <- log(XGBtesting$SalePrice)

#Create matrices from the data frames
XGBtrainingData <- as.matrix(XGBtraining, rownames.force=NA)
XGBtestingData <- as.matrix(XGBtesting, rownames.force=NA)
  
#Turn the matrices into sparse matrices
XGBtrainingSparse <- as(XGBtrainingData, "sparseMatrix")
XGBtestingSparse <- as(XGBtestingData, "sparseMatrix")
  
#####
#Cross Validate the model

#Check out the col names to then select the ones we want
colnames(XGBtrainingSparse)
vars <- c(NamesInterest1, charCols, "SalePrice")

#Convert to xgb.DMatrix format
XGBtrainingD <- xgb.DMatrix(data = XGBtrainingSparse[,vars], label = XGBtrainingSparse[,"SalePrice"]) 

#Cross validate the model
cv.sparse <- xgb.cv(data = XGBtrainingD,
                    nrounds = 600,
                    min_child_weight = 0,
                    max_depth = 10,
                    eta = 0.02,
                    subsample = .7,
                    colsample_bytree = .7,
                    booster = "gbtree",
                    eval_metric = "rmse",
                    verbose = TRUE,
                    print_every_n = 50,
                    nfold = 4,
                    nthread = 2,
                    objective="reg:linear")

#Train the model
#Choose the parameters for the model
param <- list(colsample_bytree = .7,
              subsample = .7,
              booster = "gbtree",
              max_depth = 10,
              eta = 0.02,
              eval_metric = "rmse",
              objective="reg:linear")

#Train the model using those parameters
bstSparse <-
  xgb.train(params = param,
            data = XGBtrainingD,
            nrounds = 600,
            watchlist = list(train = XGBtrainingD),
            verbose = TRUE,
            print_every_n = 50,
            nthread = 2)

#Convert Testing Sparse to xgb.DMatrix format
XGBtestingD <- xgb.DMatrix(data = XGBtestingSparse[,vars])

#Column names must match the inputs EXACTLY
#Make the prediction based on the half of the training data set aside
prediction_XGB <- predict(bstSparse, XGBtestingD)

#RMSE
rmse_XGB <- rmse(log(jTesting$SalePrice), log(prediction_XGB))

#Add to tally
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model = "XGBoost",
                                     "Log RMSE" = rmse_XGB ))

#Make table pretty
rmse_results %>% knitr::kable()

###Validate data



###Save Environment Variables
save.image(file='CapstoneHousePrices.RData')

