# import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder           # for converting non numeric data to categorical data label encoding
from sklearn.model_selection import train_test_split        #for train test splitting
from sklearn.tree import DecisionTreeClassifier as DT       #for checking testing results by Decision Tree model
from sklearn.ensemble import RandomForestClassifier         # for checking testing results by Random Forest model
from sklearn.metrics import accuracy_score, confusion_matrix   # to find the accuracy of the model and to draw confusion matrix
from sklearn.model_selection import GridSearchCV             # Hyperparamete tuning
# load the data 

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Decision Tree\datasets\Company_Data.csv")

# Univariate analysis 
data.head
data.columns
data.info()
data.isna().sum()     # No null values presents in the datasets

# descriptive statistics for each column.
data.describe()

# convert the non numeric data to numeric data 

label_encoder = LabelEncoder()

data['ShelveLoc'] = label_encoder.fit_transform(data['ShelveLoc'])
data['Urban'] = label_encoder.fit_transform(data['Urban'])
data['US'] = label_encoder.fit_transform(data['US'])

# converting the sales column from contineous to categorical data

data['sales_group'] = pd.cut(data["Sales"], bins=[-5,5,10,17], labels = ["less","medium","high"], ordered = True)

# dropping the sales column 

data.drop(["Sales"], axis=1, inplace = True)

data.sales_group.isnull().sum()

data["sales_group"].unique()
data["sales_group"].value_counts()
colnames = list(data.columns)

predictors = colnames[0:10]
target  = colnames[10]

# splitting the data into training and testing datasets 

train, test = train_test_split(data, test_size = 0.2)

# initialising the decision tree model 

model = DT(criterion = 'entropy')
# fitting the model in train datasets
model.fit(train[predictors], train[target])

# Prediction on test datasets

pred_test = model.predict(test[predictors])
pd.crosstab(test[target], pred_test, rownames = ['Actual'], colnames= ['Predictions'])
np.mean(pred_test == test[target])  # test data accuracy

# prediction on train data

pred_train = model.predict(train[predictors])
pd.crosstab(train[target], pred_train, rownames =['Actual'], colnames =['Predictions'])
np.mean(pred_train == train[target])    # Train data accuracy

# as train_accuracy > test_accuracy, this is the model is overfit 

# Lets check the accuracy of the model by Random forest model
 
rf_clf = RandomForestClassifier(n_estimators = 50, n_jobs =1, random_state = 50)
rf_clf.fit(train[predictors], train[target])

# evaluation of the model on test data

pred_rf_test = rf_clf.predict(test[predictors])
confusion_matrix(test[target], pred_rf_test)
np.mean( test[target] == pred_rf_test)

# Evaluation on training data
pred_rf_train = rf_clf.predict(train[predictors])
confusion_matrix(train[target], pred_rf_train)
np.mean(train[target] == pred_rf_train)

# again the model is overfit model 

# gridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state= 50)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = 1, cv = 5, scoring = 'accuracy')

grid_search.fit(train[predictors], train[target])

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(test[target], cv_rf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rf_clf_grid.predict(test[predictors]))

# Evaluation on Training Data
confusion_matrix(train[target], cv_rf_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_rf_clf_grid.predict(train[predictors]))


######################################################################################

# Q2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# load the data 

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Decision Tree\datasets\Diabetes.csv")

# univarialte analysis
data.shape
data.columns
data.info()
data.isna().sum()
decribe = data.describe()

# standardize the data 
 from sklearn.preprocessing import StandardScaler
 scaler = StandardScaler() 
 data_norm = scaler.fit_transform(data.iloc[:,0:8])

data['Class'].unique()
data['Class'].value_counts()

# splitting the data into train and test data 

X_train, X_test, Y_train, Y_test = train_test_split(data_norm, data.Class, test_size = 0.2)

# initialising the decision tree model

model = DT(criterion = "entropy")
model.fit(X_train,Y_train)

# evaluation the accuracy on the test dataset

pred_test = model.predict(X_test)
pd.crosstab(Y_test, pred_test, rownames = ["Actual"] , colnames = ["Predictions"])
accuracy_score(Y_test, pred_test)

# Evaluation the train data
 pred_train = model.predict(X_train)
pd.crosstab(Y_train, pred_train, rownames =["Actual"], colnames =["predictions"])
accuracy_score(Y_train, pred_train)

# The model is overfit model 

# Random Forest Classifier
 # Initialise the model
 rf = RandomForestClassifier(n_estimators = 6 , n_jobs =1, random_state =30)
 
 rf.fit(X_train, Y_train)
 # evaluation the accuracy on the test dataset
accuracy_score(Y_test, rf.predict(X_test))
pd.crosstab(Y_test, rf.predict(X_test))

# Evaluation the model on train model
accuracy_score(Y_train, rf.predict(X_train))
pd.crosstab(Y_train, rf.predict(X_train))

######
# Hyperparameter tunning with  GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=50)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, Y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

# prediction on test data
pd.crosstab(Y_test, cv_rf_clf_grid.predict(X_test))
accuracy_score(Y_test, cv_rf_clf_grid.predict(X_test))

# prediction on train data

pd.crosstab(Y_train, cv_rf_clf_grid.predict(X_train))
accuracy_score(Y_train, cv_rf_clf_grid.predict(X_train))

# As the training datasets having the accuracy of 96% > test accuracy (86%) the model is overfit

###########################################################################

# Q3)
# import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder           # for converting non numeric data to categorical data label encoding
from sklearn.model_selection import train_test_split        #for train test splitting
from sklearn.tree import DecisionTreeClassifier as DT       #for checking testing results
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# load the data 

data= pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Decision Tree\datasets\Fraud_check.csv")
# Univariate analysis 
data.head
data.columns
data.info()
data.isna().sum()     # No null values presents in the datasets

# descriptive statistics for each column.
data.describe()

# convert the non numeric data to numeric data 

data = pd.get_dummies(data,columns = ['Undergrad','Marital.Status','Urban'], drop_first = True)

# converting the Taxable.Income column from contineous to categorical data

data['TaxIncome'] = pd.cut(data["Taxable.Income"], bins=[0,30000,1000000], labels = ["Risky","Good"], ordered= True)
data.drop(["Taxable.Income"],axis=1, inplace=True)

data["TaxIncome"].unique()
data["TaxIncome"].value_counts()


from sklearn.preprocessing import StandardScaler
 scaler = StandardScaler() 
 data_norm = scaler.fit_transform(data.iloc[:,0:6])

# splitting the data into train and test data 

X_train, X_test, Y_train, Y_test = train_test_split(data_norm, data.TaxIncome, test_size = 0.2)

# initialising the decision tree model

model = DT(criterion = "entropy")
model.fit(X_train,Y_train)

# evaluation the accuracy on the test dataset

pred_test = model.predict(X_test)
pd.crosstab(Y_test, pred_test, rownames = ["Actual"] , colnames = ["Predictions"])
accuracy_score(Y_test, pred_test)

# Evaluation the train data
 pred_train = model.predict(X_train)
pd.crosstab(Y_train, pred_train, rownames =["Actual"], colnames =["predictions"])
accuracy_score(Y_train, pred_train)

# The model is overfit model 

# Random Forest Classifier
 # Initialise the model
 rf = RandomForestClassifier(n_estimators = 100 , n_jobs =1, random_state =50)
 
 rf.fit(X_train, Y_train)
 # evaluation the accuracy on the test dataset
accuracy_score(Y_test, rf.predict(X_test))
pd.crosstab(Y_test, rf.predict(X_test))

# Evaluation the model on train model
accuracy_score(Y_train, rf.predict(X_train))
pd.crosstab(Y_train, rf.predict(X_train))

######
# Hyperparameter tunning with  GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=50)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, Y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

# prediction on test data
pd.crosstab(Y_test, cv_rf_clf_grid.predict(X_test))
accuracy_score(Y_test, cv_rf_clf_grid.predict(X_test))

# prediction on train data

pd.crosstab(Y_train, cv_rf_clf_grid.predict(X_train))
accuracy_score(Y_train, cv_rf_clf_grid.predict(X_train))

# As the training datasets having the accuracy of 96% > test accuracy (86%) the model is overfit

###########################################################################

# Q4)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# load the data 

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Decision Tree\datasets\HR_DT.csv")

# univarialte analysis
data.shape
data.columns
data.info()
data.isna().sum()
decribe = data.describe()
data = data.rename(columns = {"Position of the employee" :"position", "no of Years of Experience of employee":"experience"," monthly income of employee":"Income"})

data.Income.describe()

data["Income"] =pd.cut(data["Income"], bins= [35000,60000,90000,140000], labels =["less", "good","high"])

from sklearn.preprocessing import LabelEncoder

lb= LabelEncoder()

data["position"] = lb.fit_transform(data["position"])


# standardize the data 

 from sklearn.preprocessing import StandardScaler
 scaler = StandardScaler() 
 data_norm = scaler.fit_transform(data.iloc[:,0:2])

data['Income'].unique()
data['Income'].value_counts()

# splitting the data into train and test data 

X_train, X_test, Y_train, Y_test = train_test_split(data_norm, data.Income, test_size = 0.2)

# initialising the decision tree model

model = DT(criterion = "entropy")
model.fit(X_train,Y_train)

# evaluation the accuracy on the test dataset

pred_test = model.predict(X_test)
pd.crosstab(Y_test, pred_test, rownames = ["Actual"] , colnames = ["Predictions"])
accuracy_score(Y_test, pred_test)

# Evaluation the train data
 pred_train = model.predict(X_train)
pd.crosstab(Y_train, pred_train, rownames =["Actual"], colnames =["predictions"])
accuracy_score(Y_train, pred_train)

# The model is overfit model 

# Random Forest Classifier
 # Initialise the model
 rf = RandomForestClassifier(n_estimators = 100, max_depth=10, min_samples_split=20 , n_jobs =1, random_state =30)
 
 rf.fit(X_train, Y_train)
 # evaluation the accuracy on the test dataset
accuracy_score(Y_test, rf.predict(X_test))
pd.crosstab(Y_test, rf.predict(X_test))

# Evaluation the model on train model
accuracy_score(Y_train, rf.predict(X_train))
pd.crosstab(Y_train, rf.predict(X_train))

######
# Hyperparameter tunning with  GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=50,max_depth =10, min_samples_split = 20, n_jobs=1, random_state=50)

param_grid = {"max_features": [1,2,3], "min_samples_split": [3]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, Y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

# prediction on test data
pd.crosstab(Y_test, cv_rf_clf_grid.predict(X_test))
accuracy_score(Y_test, cv_rf_clf_grid.predict(X_test))

# prediction on train data

pd.crosstab(Y_train, cv_rf_clf_grid.predict(X_train))
accuracy_score(Y_train, cv_rf_clf_grid.predict(X_train))

# As the training datasets having the accuracy of 98% > test accuracy (90%) the model is overfit
