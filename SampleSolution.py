import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

#test size constant
TS=0.20

"""Features of the dataset are: Feature 0 - age, Feature1 - sex, Feature2 - bmi, Feature3 - map, Feature4 - tc, Feature5 - ldl, Feature6 - hdl, Feature7 - tch, Feature8 - ltg, Feature9 - glu.
Splitted the data here. TS is the constant for initiating test size of this splitt. 
"""

#function fot spliting train and test sets
def data_split(X,y,TS):  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS, random_state=1)
  return X_train, X_test, y_train, y_test

"""Used Mutual Information for Feature selection."""

#function for feature selection
def feature_selection(X_train,y_train,X_test):
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

"""From the alalysis came to know that feature0 (age) and feature5 (ldl) is less important feature among all the feature. Output is displayed below. Thats why I excluded this 2 features from the model. """

def feature_output(fs):
  print("\n")
  print("#########Feature Selection########")
  for i in range(len(fs.scores_)):
	  print('Feature %d: %f' % (i, fs.scores_[i]))
  # plot the scores
  pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
  pyplot.title("Feature Selection")
  pyplot.xlabel("Features")
  pyplot.ylabel("Mutual Information")
  pyplot.show()

"""I have used Linear Regression and trained 8 of the features using this. I have also calculated R square value, Mean Absolute Error, Mean Squeared Error and Root Mean Squared Error For performance evaluation. """

def linear_regression(X_train,y_train,X_test,y_test):
  model=LinearRegression()
  model.fit(X_train,y_train)
  y_predicted=model.predict(X_test)
  r_sq = model.score(X_train,y_train)
  print("\n")
  print("#########Linear Regression########")
  print('R Square:', r_sq)
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predicted))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predicted))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
  return y_predicted

def linear_regression_output(X_test,y_test,y_predicted):
  pyplot.scatter(y_predicted,y_test)
  pyplot.title("Linear Regression")
  pyplot.xlabel("Predicted")
  pyplot.ylabel("Actual")
  figure2=pyplot.figure(2)
  figure2.show(2)

"""I have used Random Forest Regressor and trained 8 of the features using this. I have also calculated R square value, Mean Absolute Error, Mean Squeared Error and Root Mean Squared Error For performance evaluation."""

def random_forest(X_train,y_train,X_test,y_test):
  model=RandomForestRegressor(random_state=0)
  model.fit(X_train,y_train)
  y_predicted=model.predict(X_test)
  r_sq = model.score(X_train,y_train)
  print("\n")
  print("#########Random Forest Regressor########")
  print('R Square:', r_sq)
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predicted))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predicted))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
  return y_predicted

def rf_regressor_output(X_test,y_test,y_pred):
  pyplot.scatter(y_pred,y_test)
  pyplot.title("Random Forest Regressor")
  pyplot.xlabel("Predicted")
  pyplot.ylabel("Actual")
  figure3=pyplot.figure(3)
  figure3.show()

def main(): 
  #loading dataset
  X,y = load_diabetes(return_X_y = True) 
  
  #splitting dataset into train and test
  X_train, X_test, y_train, y_test = data_split(X,y,TS) 

  #feature selection
  X_train_fs, X_test_fs, fs = feature_selection(X_train, y_train, X_test) 

  #feature_selection_output
  feature_output(fs)
  
  #setting features
  X_train=X_train[:,[1,2,3,4,6,7,8,9]]
  X_test=X_test[:,[1,2,3,4,6,7,8,9]]
  #Linear regression
  y_predicted=linear_regression(X_train,y_train,X_test,y_test)    

  #Linear Regression output
  linear_regression_output(X_test,y_test,y_predicted)    

  #random forest
  y_pred=random_forest(X_train,y_train,X_test,y_test)

  #random forest regressor output
  rf_regressor_output(X_test,y_test,y_pred)

"""Outcome: I have used linear regression and radom forest regressor on the data set. 
I have excluded 2 of the features among 10 features because I got them less important from mutual information feature selection analysis.
 Calculated  R square value, Mean Absolute Error, Mean Squeared Error and Root Mean Squared Error for both of the model. 
 And it is observed that Random Forest Regressor have higher  R square , Mean Absolute Error, Mean Squeared Error and Root Mean Squared Error.
 I have also represented both of the model in figures. """

if __name__ == "__main__":
    main()