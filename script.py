# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 14:02:51 2019

@author: Pari
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import itertools


#%matplotlib inline

data_paths = {
         "trentino_grid": r"input\1_Trentino_Grid\trentino-grid.geojson",
         "internet": r"input\2_Telecommunications_Internet\Internet_traffic_activity.csv",
         "precipitation": r"input\3_Precipitation_-_Trentino\precipitation-trentino-201312.csv",
         "precipitation_availability": r"input\3_Precipitation_-_Trentino\precipitation-trentino-data-availability-201312.csv",
         "electricity_line": r"input\4_SET_Electricity\line.csv",
         "electricity_dec": r"input\4_SET_Electricity\SET-dec-2013.csv",
         "air_quality": r"input\5_Air_Quality_TN\air-2013-12.csv"
         }

 
df_trentino = gpd.read_file(data_paths["trentino_grid"])
df_trentino.set_index('cellId', inplace=True)
ax = df_trentino.plot()

df_internet = pd.read_csv(data_paths["internet"],
                          header=None,
                          names=['Square_id',
                                 'Timestamp',
                                 'Internet_traffic_activity'
                                 ]
                          )

df_internet.head()
print(len(df_internet[df_internet['Square_id'] == 38]) / 24)
df_internet.drop_duplicates(inplace=True)
df_internet['Timestamp'] = pd.to_datetime(df_internet['Timestamp'])


df_internet['day'] = df_internet['Timestamp'].dt.day

day_wise_internet_use = df_internet.groupby('day')['Internet_traffic_activity'].mean()

f = plt.figure()
ax = day_wise_internet_use.plot()
plt.xlabel("Day")
plt.ylabel("Average Internet Traffic Activity")



df_internet['hour'] = df_internet['Timestamp'].dt.hour
hour_wise_internet_use = df_internet.groupby('hour')['Internet_traffic_activity'].mean()
f = plt.bar(hour_wise_internet_use.index, hour_wise_internet_use)
plt.xlabel("Hour")
plt.ylabel("Average Internet Traffic Activity")


out = df_internet.groupby("Square_id").agg({'Internet_traffic_activity': 'mean'})

df_geo_internet = pd.merge(df_trentino,
                             out,
                             how='inner',
                             left_index=True,
                             right_index=True
                             )


# set a variable that will call whatever column we want to visualise on the map
variable = 'Internet_traffic_activity'
#
## set the range for the choropleth
#vmin, vmax = 0, 1320

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.axis('off')
ax.set_title('Average Internet usage in December 2013 in Trentino',
             fontdict={'fontsize': '20',
                       'fontweight' : '3'
                       }
             )
#sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
### empty array for the data range
#sm._A = []
### add the colorbar to the figure
#cbar = fig.colorbar(sm)


df_geo_internet.plot(column=variable,
                     cmap='Blues',
                     linewidth=0.8,
                     ax=ax,
                     edgecolor='0.8',
                     k=8,
                     scheme='Quantiles',
                     legend=True
                     )

ax.get_legend().set_bbox_to_anchor((.01, .4))


#NR_UBICAZIONI == Numberofcostumersites

df_electricity_line = pd.read_csv(data_paths["electricity_line"])
df_electricity_usage = pd.read_csv(data_paths["electricity_dec"],
                                   header=None)

df_electricity_usage.describe()


df_electricity_usage['date'] = pd.to_datetime(df_electricity_usage[1])
#
#target = df_electricity_usage.groupby(0).resample('3H', on='date').mean().reset_index()
#print(target.head())

#Expected all squareid to be unique, but there are repetitions.
df_electricity_line['SQUAREID'].nunique()
df_electricity_line.drop_duplicates() #did nothing

df_electricity_line[df_electricity_line['SQUAREID'] == 860]


idx = pd.date_range('12-01-2013', '12-31-2013 23:00:00', freq='H')
def interpolate_for_missing_time(df):
    if len(df) == len(idx):
        return df['Internet_traffic_activity']
    temp = df['Internet_traffic_activity']
    temp.index = pd.DatetimeIndex(df['Timestamp'])
    temp = temp.reindex(idx, fill_value=np.nan)
    temp = temp.interpolate()
    return temp 

out = df_internet.groupby('Square_id').apply(interpolate_for_missing_time).reset_index()
out.groupby('Square_id')

X = out['Internet_traffic_activity'].values.reshape(-1, 31, 8, 3)
X = X.mean(axis=-1)

square_ids = out['Square_id'].unique()
days = np.arange(1, 32)
time_chunks = np.arange(1, 9)

to_join = pd.DataFrame(itertools.product(square_ids, days),
                    columns=['Square_id', 'day'])

temp = pd.DataFrame(X.reshape(-1, len(time_chunks)), columns=time_chunks)

to_join = pd.concat([to_join, temp], axis=1)

electricity_by_square = pd.merge(df_electricity_line,
                                 to_join, 
                                 how='inner',
                                 left_on='SQUAREID', 
                                 right_on='Square_id'
                                 )


agg_func = {k: 'sum' for k in time_chunks}
agg_func.update({'SQUAREID': 'count',
                 'NR_UBICAZIONI': 'sum',
                 }
                )
final_df = electricity_by_square.groupby(['LINESET', 'day']).agg(agg_func)

target = df_electricity_usage.copy()
target['day'] = target['date'].dt.day
target.rename(columns={0:'LINESET',
                       2: 'Amperes'},
                inplace=True)



out = target.groupby(['LINESET', 'day']).agg({'Amperes': 'max'})

df = out.merge(final_df, left_index=True, right_index=True)

X = df[time_chunks]
Y = df['Amperes']


from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.25,
                                                    random_state=42
                                                    )

scaler = preprocessing.StandardScaler().fit(X_train)
X_tr_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = linear_model.LinearRegression()
lr.fit(X_tr_scaled, y_train)
y_pred_tr = lr.predict(X_tr_scaled)
y_pred = lr.predict(X_test_scaled)


print('Coefficients: \n', lr.coef_)
# The mean squared error
print("Mean squared error Train: {:.2f}".format(mean_squared_error(y_train, y_pred_tr)))
print("Mean squared error Test: {:.2f}".format(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


tree_regr = DecisionTreeRegressor(random_state=42)

tree_regr.fit(X_tr_scaled, y_train)
y_pred_tr = tree_regr.predict(X_tr_scaled)
y_pred = tree_regr.predict(X_test_scaled)


print('Feature Importances: \n', tree_regr.feature_importances_)
# The mean squared error
print("Mean squared error Train: {:.2f}".format(mean_squared_error(y_train, y_pred_tr)))
print("Mean squared error Test: {:.2f}".format(mean_squared_error(y_test, y_pred)))




