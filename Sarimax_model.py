import os.path
from pathlib import Path
import pandas as pd
from clearml import Dataset, Task, OutputModel
import global_config
import numpy as np
from scipy import stats
import seaborn as sns

import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.plotting import lag_plot
from statsmodels.graphics.api import qqplot
#%matplotlib inline

from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import itertools
from statsmodels.tsa.arima_model import ARIMA


from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='Sarimax_model_training',
    output_uri=True
)

# Set default docker
#task.set_base_docker(docker_image="python:3.7")

# Training args
training_args = {
    'p':5,
    'd':0,
    'q':8,
    'p1':1,
    'd1':0,
    'q1':1,
    's':12,
    'dataset_id':'f80ac9a8e8114852ad1049dcfe42227c'
    
}
task.connect(training_args)

# Load our Dataset
dataset = Dataset.get(dataset_id='f80ac9a8e8114852ad1049dcfe42227c'
    #dataset_name='preprocessed_sales_dataset',
    #dataset_project=global_config.PROJECT_NAME
)
local_folder = dataset.get_local_copy()

df= pd.read_csv(Path( local_folder)/ 'MISSING_VALUE_HANDLED.csv')

df.info()

df=df.sort_values(by='Date') 
df

df= df[df['Store'] == 2]
df=df[["Date","Sales"]]
df['Date']=pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)
df.head()

df.tail()
df.describe()
df.plot()
plt.title("sales of store")
plt.show()


decomposition = sm.tsa.seasonal_decompose(df,model='additive', period=5)
decomp = decomposition.plot()
decomp.suptitle('"A" Value Decomposition')
plt.tight_layout()
plt.show()

df1=df.loc['2013-01-01':'2013-12-31']
df1.plot(figsize=(20,8))
plt.title("2013 sales")
plt.show()

df2=df.loc['2014-01-01':'2014-12-31']
df2.plot(figsize=(20,8))
#df2.plot(figsize=(20,8))
plt.title("2014 sales")
plt.show()

df.shape

df.hist()
df.plot(kind='kde')
plt.title("Distribution of sales plot")
plt.show()

lag_plot(df)
plt.title("lag sales plot")
plt.show()

moving_average_df=df.rolling(window=20).mean()  
moving_average_df

moving_average_df.plot()
plt.title("Moving avrage of sales plot")
plt.show()

sm.stats.durbin_watson(df) # correlation

# acf and pacf plots
fig = sm.graphics.tsa.plot_acf(df.values.squeeze(), lags=40)
plt.show()

#ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df, lags=40)
plt.show()


# Test for stationarity
test_result=adfuller(df['Sales'])
test_result

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    print('Critical Values:')
    for key, value in result[4].items():
        print(key, value)
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    
adfuller_test(df['Sales'])

#test data 2015-07-01 to 2015-07-31
training_data,test_data=df[df.index<'2015-07-01'],df[df.index>='2015-07-01']

print(df.shape)
print(training_data.shape)
print(test_data.shape)

# SARIMA
p = d = q = range(0, 2) # Define the p, d and q parameters to take any value between 0 and 2

pdq = list(itertools.product(p, d, q)) # Generate all different combinations of p, q and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(training_args['p'],training_args['d'],training_args['q']),seasonal_order=(training_args['p1'],training_args['d1'],training_args['q1'],training_args['s']))
results=model.fit()
results.summary()

test_data['forecast']=results.predict(start=-31,dynamic=True)
test_data[['Sales','forecast']].plot(figsize=(12,8))
plt.show()

print(test_data)

print("RMSE =",np.sqrt(mean_squared_error(test_data['Sales'], test_data['forecast'])))

rmspe = np.sqrt(mean_squared_error(test_data['Sales'], test_data['forecast']) / np.mean(test_data['Sales']))

print(rmspe)

#forecast = results.forecast(steps=42)
#print(forecast)

test_data1=pd.DataFrame()
test_data1["forecast"] = results.forecast(steps=36)
print(test_data1)


data = test_data[["Sales","forecast",]].append(test_data1[["forecast"]])

#data[["Sales","forecast"]].plot(figsize=(12,8))
#plt.title("next six weeks forecasted sale")
#plt.show()

#forecast.plot(figsize=(12,8))

#plt.show()

plt.plot(data["Sales"],label="Actual Sales")
plt.plot(data["forecast"],label="forecast Sales")
plt.title("Next six weeks forecasted sale")
plt.legend()
plt.show()

task.get_logger().report_scalar(
    title='Performance',
    series='RMSPE',
    value=rmspe,
    iteration=0
)

results.save("Sarimax.pkl")

task.upload_artifact(name='sarimax_model', artifact_object='Sarimax.pkl')

task.close()