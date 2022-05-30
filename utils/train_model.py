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
df_train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

# Fit model
# Our forest consists of 200 trees with a max depth of 8 
RF = RandomForestRegressor(n_estimators=200, max_depth=8)
# Fitting the model
RF.fit(X_train,y_train)
RF_predict = RF.predict(X_test)

#saving the pickle file
RF_save_path = "RF.pkl"
with open(RF_save_path,'wb') as file:
    pickle.dump(RF,file)
    
#en_load_path = "v_reg.pkl"
RF_load_path = "RF.pkl"
with open(RF_load_path,'rb') as file:
    unpickled_RF = pickle.load(file)    

RF_predict =  unpickled_RF.predict(X)

df_new = pd.DataFrame(RF_predict, columns=['load_shortfall_3h'])
df_test.head()
Unnamed: 0	time	Madrid_wind_speed	Valencia_wind_deg	Bilbao_rain_1h	Valencia_wind_speed	Seville_humidity	Madrid_humidity	Bilbao_clouds_all	Bilbao_wind_speed	...	Barcelona_temp_max	Madrid_temp_max	Barcelona_temp	Bilbao_temp_min	Bilbao_temp	Barcelona_temp_min	Bilbao_temp_max	Seville_temp_min	Madrid_temp	Madrid_temp_min
0	8763	2018-01-01 00:00:00	5.000000	level_8	0.0	5.000000	87.000000	71.333333	20.000000	3.000000	...	287.816667	280.816667	287.356667	276.150000	280.380000	286.816667	285.150000	283.150000	279.866667	279.150000
1	8764	2018-01-01 03:00:00	4.666667	level_8	0.0	5.333333	89.000000	78.000000	0.000000	3.666667	...	284.816667	280.483333	284.190000	277.816667	281.010000	283.483333	284.150000	281.150000	279.193333	278.150000
2	8765	2018-01-01 06:00:00	2.333333	level_7	0.0	5.000000	89.000000	89.666667	0.000000	2.333333	...	284.483333	276.483333	283.150000	276.816667	279.196667	281.816667	282.150000	280.483333	276.340000	276.150000
3	8766	2018-01-01 09:00:00	2.666667	level_7	0.0	5.333333	93.333333	82.666667	26.666667	5.666667	...	284.150000	277.150000	283.190000	279.150000	281.740000	282.150000	284.483333	279.150000	275.953333	274.483333
4	8767	2018-01-01 12:00:00	4.000000	level_7	0.0	8.666667	65.333333	64.000000	26.666667	10.666667	...	287.483333	281.150000	286.816667	281.816667	284.116667	286.150000	286.816667	284.483333	280.686667	280.150000
5 rows × 48 columns

output_en_df = pd.DataFrame({"time": df_test_copy['time'].reset_index(drop=True)})
en_file = output_en_df.join(df_new)
en_file['load_shortfall_3h'] = df_new
en_file.to_csv("RF_model_file.csv", index=False)
print(en_file)
                     time  load_shortfall_3h
0     2018-01-01 00:00:00        9454.853081
1     2018-01-01 03:00:00        9334.159023
2     2018-01-01 06:00:00        9352.296410
3     2018-01-01 09:00:00        9352.296410
4     2018-01-01 12:00:00        9451.513864
...                   ...                ...
2915  2018-12-31 09:00:00        4975.373938
2916  2018-12-31 12:00:00        8648.271280
2917  2018-12-31 15:00:00        8447.310486
2918  2018-12-31 18:00:00        8109.908496
2919  2018-12-31 21:00:00        7488.794840

[2920 rows x 2 columns]
output = pd.DataFrame({"time": df_test_copy['time']})
submission = output.join(df_new)
submission.to_csv('new_submission.csv',index = False)

submission.head()
