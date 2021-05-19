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
#Reduced the changes to 5 variables - see a bit later
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


##############################################################
#REMEMBER TO Do THE FOLLOWING IN THE TESTING AND FINAL SET
##############################################################

#Let's convert those to factors and then run one more set of correlations
#GarageType
price <- jTraining %>%
  group_by(GarageType) %>%
  summarize(avg=mean(SalePrice, na.rm=T)) %>%
  arrange(desc(avg))

jTraining$jGarageType[jTraining$GarageType %in% c("BuiltIn", "Attchd")] <- 3
jTraining$jGarageType[jTraining$GarageType %in% c("Basment", "Detchd", "2Types")] <- 2
jTraining$jGarageType[jTraining$GarageType %in% c("CarPort", "NoGarage")] <- 1

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

#MSZoning
price <- jTraining %>%
        group_by(MSZoning) %>%
        summarize(avg=mean(SalePrice, na.rm=T)) %>%
        arrange(desc(avg))

jTraining$jMSZoning[jTraining$MSZoning == "FV"] <- 4
jTraining$jMSZoning[jTraining$MSZoning == "RL"] <- 3
jTraining$jMSZoning[jTraining$MSZoning %in% c("RH","RM")] <- 2
jTraining$jMSZoning[jTraining$MSZoning == "C (all)"] <- 1

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

#Look at the correlations for the character variables
charCols <- c("jGarageType", "jKitchenQual", "jExterQual", "jMSZoning", "jNeighborhood", "SalePrice")
correlations <- cor(jTraining[, charCols],use="everything")
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
save.image(file='CapstoneHousePrices.RData')

