rm(list=ls(all=TRUE))
setwd("")
setwd("C:/Users/Lenovo/Desktop/Machine learning/SVM_Lab05/20181203_Batch 34_CSE 7305c_SVM_Lab05")
### Import data into R - "UniversalBank.csv"
Universal<-read.csv("UniversalBank.csv",header=T,sep = ",")

#OBJECTIVE: Will a person take a personal loan or not? 
#Response variable is "Personal.Loan" 

# Understanding the data 
str(Universal)
summary(Universal)

# Removing unnecessary columns ID and zipcode
Universal$ID<-NULL
Universal$ZIP.Code<-NULL

# Do necessary type conversions
Universal$Education<-as.factor(Universal$Education)
Universal$Personal.Loan<-as.factor(Universal$Personal.Loan)
Universal$Securities.Account<-as.factor(Universal$Securities.Account)
Universal$CD.Account<-as.factor(Universal$CD.Account)
Universal$Online<-as.factor(Universal$Online)
Universal$CreditCard<-as.factor(Universal$CreditCard)



# Do Train-Test Split
library(caret)
set.seed(123)
train_rows<-createDataPartition(Universal$Personal.Loan,p=0.7,list = F)
train<-Universal[train_rows,]
test<-Universal[-train_rows,]

# PreProcess the data to standadize the numeric attributes
preProc<-preProcess(train[,setdiff(names(train),"Personal.Loan")],method = c("center", "scale"))
train<-predict(preProc,train)
test<-predict(preProc,test)

###create dummies for factor varibales 
dummies <- dummyVars(Personal.Loan~.,data=Universal)

x.train=predict(dummies, newdata = train)
y.train=train$Personal.Loan
x.test = predict(dummies, newdata = test)
y.test = test$Personal.Loan

####Classification using "e1071"####
# install.packages("e1071")
library(e1071)


# Building the model on train data
model  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "linear", cost = 10)
summary(model)

#The "cost" parameter balances the trade-off between having a large margin and classifying
#all points correctly. It is important to choose it well to have good
#generalization.

# Predict on train and test using the model
pred_train<-predict(model,x.train)
pred_test<-predict(model,x.test)
# Build Confusion matrix
confusionMatrix(pred_train,y.train)
confusionMatrix(pred_test,y.test)

#######Build SVM model with RBF kernel#### 
model_RBF = svm(x.train,y.train, method = "C-classification", kernel = "radial", cost = 10,
            gamma = 0.1)
summary(model_RBF)

# Predict on train and test using the model
pred_train1<-predict(model_RBF,x.train)
pred_test1<-predict(model_RBF,x.test)

# Build Confusion matrix
confusionMatrix(pred_train1,y.train)
confusionMatrix(pred_test1,y.test)

#####Classification using "KSVM"############
#install.packages("kernlab")
library(kernlab)

#Build model using ksvm with "rbfdot" kernel
kern_rbf <- ksvm(x.train,y.train,
                  type='C-svc',kernel="rbfdot",kpar="automatic",
                  C=10, cross=5)
kern_rbf

# Predict on train and test using the model
pred_train2<-predict(kern_rbf,x.train)
pred_test2<-predict(kern_rbf,x.test)

# Build Confusion matrix
confusionMatrix(pred_train1,y.train)
confusionMatrix(pred_test1,y.test)

#Build model using ksvm with "vanilladot" kernel
kern_vanilla <- ksvm(x.train,y.train,
                     type='C-svc',kernel="vanilladot", C = 10)
kern_vanilla

# Predict on train and test using the model

# Build Confusion matrix


#Grid Search/Hyper-parameter tuning

tuneResult <- tune(svm, train.x = x.train, train.y = y.train, 
                   ranges = list(gamma = 10^(-3:-1), cost = 2^(2:3)),class.weights= c("0" = 1, "1" = 10),tunecontrol=tune.control(cross=3))
print(tuneResult) 
summary(tuneResult)

#Predict model and calculate errors
tunedModel <- tuneResult$best.model;tunedModel

# Predict on train and test using the model

# Build Confusion matrix

