"""
    Helper functions for the pretrained model to be used within our API.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------
    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  
"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.
    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.
    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
   #Filling in the missing values with the mean

df_train = pd.read_csv('df_train.csv') # load the train data
df_test = pd.read_csv('df_test.csv')
df_train['Valencia_pressure'].fillna(df_train['Valencia_pressure'].mean(), inplace = True)
#converting Valencia_wind_deg and Seville_pressure columns from categorical to numerical datatypes.

df_train['Valencia_wind_deg'] = df_train['Valencia_wind_deg'].str.extract('(\d+)').astype('int64')
df_train['Seville_pressure'] = df_train['Seville_pressure'].str.extract('(\d+)').astype('int64')
#The next step is to engineer new features from the time column

#Engineering New Features ( i.e Desampling the Time) to further expand our training data set

df_train['Year']  = df_train['time'].astype('datetime64').dt.year
df_train['Month_of_year']  = df_train['time'].astype('datetime64').dt.month
df_train['Week_of_year'] = df_train['time'].astype('datetime64').dt.weekofyear
df_train['Day_of_year']  = df_train['time'].astype('datetime64').dt.dayofyear
df_train['Day_of_month']  = df_train['time'].astype('datetime64').dt.day
df_train['Day_of_week'] = df_train['time'].astype('datetime64').dt.dayofweek
df_train['Hour_of_week'] = ((df_train['time'].astype('datetime64').dt.dayofweek) * 24 + 24) - (24 - df_train['time'].astype('datetime64').dt.hour)
df_train['Hour_of_day']  = df_train['time'].astype('datetime64').dt.hour
#Let us have a look at the correlation(s) between our newly created temporal features

Time_df = df_train.iloc[:,[-8,-7,-6,-5,-4,-3,-2,-1]]
plt.figure(figsize=[10,6])
sns.heatmap(Time_df.corr(),annot=True )
#<AxesSubplot:>

##Looking at our heatmap tells us that we have high Multicollinearity present in our new features. The features involved are -

#Week of the year. Day of the year. Month of the year. Day of the week. Hour of the week. We would have to drop one of the features that have high correlation with each other.

#Alongside dropping these features mentioned above, we would also be dropping the time and Unnamed column.

df_train = df_train.drop(columns=['Week_of_year','Day_of_year','Hour_of_week', 'Unnamed: 0','time'])
plt.figure(figsize=[35,15])
sns.heatmap(df_train.corr(),annot=True )
#<AxesSubplot:>

#Just as we mentioned in our EDA, we noticed the presence of high correlations between the predictor columns and also possible outliers.

#Here, we would have to drop these columns to improve the performance of our model and reduce any possibility of overfitting in our model. Let us check if this approach corresponds with our feature selection. Using SelectKBest and Chi2 to perform Feature Selection.

## Splitting our data into dependent Variable and Independent Variable
X = df_train.drop(columns = 'load_shortfall_3h')
y = df_train['load_shortfall_3h'].astype('int')
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Features', 'Score']
new_X = featureScores.sort_values('Score',ascending=False).head(40)
new_X.head(40) #To get the most important features based on their score 
#Features	Score
#18	Barcelona_pressure	1.189344e+09
#9	Bilbao_wind_deg	4.574064e+05
#8	Seville_clouds_all	3.049398e+05
#11	Barcelona_wind_deg	2.920143e+05


X = X[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
       'Seville_pressure', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
       'Valencia_pressure', 'Bilbao_weather_id', 
        'Valencia_humidity', 'Year', 'Month_of_year', 'Day_of_month', 'Day_of_week', 'Hour_of_day']]
plt.figure(figsize=[20,10])
sns.heatmap(X.corr(),annot=True )
#<AxesSubplot:>



# Create standardization object
scaler = StandardScaler()
# Save standardized features into new variable
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
X_scaled.head()


y.head()
