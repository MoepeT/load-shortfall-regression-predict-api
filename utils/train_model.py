"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""
# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 #%matplotlib inline
import seaborn as sns
import plotly.express as px
from statsmodels.graphics.correlation import plot_corr


# Libraries for data preparation and model building
#from sklearn.linear_model import LinearRegrepip #install plotlyssion
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR



from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


import warnings

warnings.filterwarnings("ignore")


import pickle
# Fetch training data and preprocess for modeling
df_train = pd.read_csv('data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

#saving the pickle file
RF_save_path = "RF.pkl"
with open(RF_save_path,'wb') as file:
    pickle.dump(RF,file)
