import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def sort_dataset(dataset_df):
    
    return dataset_df.sort_values(by='year', ascending=True)

def split_dataset(dataset_df):
    X = dataset_df.drop("salary", axis=1)
    Y = dataset_df["salary"] * 0.001
    X_train, X_test, Y_train, Y_test = train_test_split(X[:1718], Y[:1718])
    
    return X_train, X_test, Y_train, Y_test


def extract_numerical_cols(dataset_df):
    nc = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']

    return dataset_df[nc]


def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, Y_train)
    
    return dt_reg.predict(X_test)


def train_predict_random_forest(X_train, Y_train, X_test):
    pipe = make_pipeline(StandardScaler(), RandomForestRegressor())
    pipe.fit(X_train, Y_train)
    
    return pipe.predict(X_test)


def train_predict_svm(X_train, Y_train, X_test):
    svm_pipe = make_pipeline(StandardScaler(), SVR())
    svm_pipe.fit(X_train, Y_train)
    
    return svm_pipe.predict(X_test)


def calculate_RMSE(labels, predictions):
    return np.sqrt(np.mean((labels - predictions) ** 2))

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
