#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[9]:


import os.path
from pathlib import Path
import  pickle
import pandas as pd
from clearml import Dataset, Task, OutputModel
import global_config

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import datetime
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline




# In[10]:

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='EDA',
    task_type='data_processing',
    reuse_last_task_id=False
)

# Get the dataset
dataset2 = Dataset.get(
    dataset_project=global_config.PROJECT_NAME,
    dataset_name='raw_sales_dataset',
    alias='my_raw_dataset2'
)
local_folder = dataset2.get_local_copy()
print(f"Using dataset ID: {dataset2.id}")

# Clean up the data a little bit
df = pd.read_csv((Path(local_folder) / 'feature_engneering.csv'))




# In[11]:
df.info()


# In[12]:
df.head()


# In[13]:
df.tail()


# In[14]:
fig, (axis1, axis2, axis3) = plt.subplots(1, 3,figsize=(10,5), gridspec_kw={'wspace': 0.5})


# First subplot
mask = df['Sales'] > 0
sns.boxplot(df[mask]['Sales'], ax=axis1)
axis1.set_title('Boxplot Sales for Open stores')
axis1.set_xlabel('Sales')
axis1.set_ylabel('')

data = df[mask]["Sales"]
#data = data.values.tolist()
axis2.hist(data, color='r', bins=30)
axis2.set_title('Histogram Sales for Open stores')
axis2.set_xlabel('Sales Value')
axis2.set_ylabel('Frequency')


# Second subplot
# Display and save the figure
data = df[mask]['Sales']
axis3.hist(x=np.log(data), range=(7, 11), bins=40)
axis3.set_title('Histogram of Log Sales for Open stores')
axis3.set_xlabel('Log Sales Value')
axis3.set_ylabel('Frequency')

plt.show()

task.get_logger().report_matplotlib_figure(title='Sales Plot', series='Open stores sales', figure=fig, 
                                            iteration=1, report_image=True, report_interactive=True)





# In[15]:
print("Median Sales:", np.median(df[mask]["Sales"]))
print("="*50)
print("75th percentile Sales:", np.quantile(df[mask]["Sales"], 0.75))
print("="*50)
for i in range(0,10):
    print('{}th percetile Sale is : {}'.format((90+i), np.quantile(df[mask]["Sales"], .9+i/100)))
print("="*50)
for i in range(0,10):
    print('{}th percetile Sale is : {}'.format((99+i/10), np.quantile(df[mask]["Sales"], .99+i/1000)))


# In[16]:
outliers = df[df["Sales"]>23979].sort_values('Sales')
outliers.shape

#x = np.arange(0,1018)
#plt.plot(x, outliers['Sales'][:1018])


# In[17]:
len(outliers)


# In[18]:
x = np.arange(0,1018)
plt.plot(x, outliers['Sales'][:1018])
plt.title("Outlier")
plt.show()


# In[19]:
print("25th percentile Sales:", np.quantile(df[mask]["Sales"], 0.25))
print("="*50)


# In[20]:
IQR=8360.0-4859.0
print(IQR)


# In[21]:
max=8360+2.5*3501
print(max)


# In[22]:
df[df['Sales']> 25000]


# In[23]:
# Plotting sales distribution plot for different stores
store_1 = df.loc[(df["Store"]==1)&(df['Sales']>0) , ['Date',"Sales"]] # Store 1
store_10 = df.loc[(df["Store"]==10)&(df['Sales']>0), ['Date',"Sales"]] # Store 10
store_600 = df.loc[(df["Store"]==600)&(df['Sales']>0) ,['Date',"Sales"]] # Store 600
store_1100 = df.loc[(df["Store"]==1100)&(df['Sales']>0) , ['Date',"Sales"]] # Store 1100
f = plt.figure(figsize=(16,12))
plt.subplots_adjust(hspace = 0.5)
ax1 = f.add_subplot(411)
ax1.plot(store_1['Date'], store_1['Sales'], '-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Sales')
ax1.set_title('Store 1 Sales Distribution')

ax2 = f.add_subplot(412)
ax2.plot(store_10['Date'], store_10['Sales'], '-')
ax2.set_xlabel('Time')
ax2.set_ylabel('Sales')
ax2.set_title('Store 10 Sales Distribution')

ax2 = f.add_subplot(413)
ax2.plot(store_600['Date'], store_600['Sales'], '-')
ax2.set_xlabel('Time')
ax2.set_ylabel('Sales')
ax2.set_title('Store 600 Sales Distribution')

ax2 = f.add_subplot(414)
ax2.plot(store_1100['Date'], store_1100['Sales'], '-')
ax2.set_xlabel('Time')
ax2.set_ylabel('Sales')
ax2.set_title('Store 1100 Sales Distribution')

plt.title("sales distribution plot for different stores")
plt.show()
     


# In[24]:
# For different stores
def plotseasonal(res, axes, title):
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')
    axes[0].set_title(title)


fig, axes1 = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(16,5))
fig, axes2 = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(16,5))

store_5 = df.loc[(df["Store"]==5)&(df['Sales']>0), ['Date',"Sales"]].set_index('Date') # Store 5
store_50 = df.loc[(df["Store"]==50)&(df['Sales']>0), ['Date',"Sales"]].set_index('Date') # Store 50
store_505 = df.loc[(df["Store"]==505)&(df['Sales']>0), ['Date',"Sales"]].set_index('Date') # Store 505
store_1055 = df.loc[(df["Store"]==1055)&(df['Sales']>0), ['Date',"Sales"]].set_index('Date') # Store 1055

result5 = seasonal_decompose(store_5, model='additive', period=30)
result50 = seasonal_decompose(store_50, model='additive', period=30)
result505 = seasonal_decompose(store_505, model='additive', period=30)
result1055 = seasonal_decompose(store_1055, model='additive', period=30)

plotseasonal(result5, axes1[:,0], title = 'Sales decomposition for Store 5')
plotseasonal(result50, axes1[:,1], title = 'Sales decomposition for Store 50')
plotseasonal(result505, axes2[:,0], title = 'Sales decomposition for Store 505')
plotseasonal(result1055, axes2[:,1], title = 'Sales decomposition for Store 1055')

plt.tight_layout()
plt.title("Decomposition of sales for different stores")
plt.show()


# In[25]:
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Sales'][500000:600000], autolag='AIC')
 #Extracting the values from the results:

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")


# In[26]:
f = plt.figure(figsize=(15,10))
plt.subplots_adjust(hspace = 0.2)
ax1 = f.add_subplot(221)
plot_acf(store_5, lags = 90, ax= ax1, title = 'Autocorrelation plot for store 5')
ax2 = f.add_subplot(222)
plot_acf(store_50, lags = 90, ax= ax2, title = 'Autocorrelation plot for store 50')
ax3 = f.add_subplot(223)
plot_acf(store_505, lags = 90, ax= ax3, title = 'Autocorrelation plot for store 505')
ax4 = f.add_subplot(224)
plot_acf(store_1055, lags = 90, ax= ax4, title = 'Autocorrelation plot for store 1055')

plt.title("Autocorrelation plots")
plt.show()


# In[27]:
df.info()


# In[28]:
# Day Of Week
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
day_names = { 0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',6: 'Sunday',}

dw_df = df.groupby('DayOfWeek')['Sales'].mean()
dw_df_c = df.groupby('DayOfWeek')['Customers'].mean()

dw_df.index = dw_df.index.map(day_names)
dw_df_c.index = dw_df_c.index.map(day_names)

# Plot the average number of sales on each day of the week
ax1.bar(dw_df.index, dw_df.values)
ax1.set_title('Average sales on a given day of the week')
ax1.set_ylabel('Sales')

# Plot the average number of customers on each day of the week
ax2.bar(dw_df_c.index, dw_df_c.values)
ax2.set_title('Average no.of customers on a given day of the week')
ax2.set_ylabel('Customers')

fig.suptitle("Average Sales and customer on given day of week for open and close store")
plt.show()

fig, (ax3,ax4) = plt.subplots(1,2,figsize=(14,5))

dw_m = df[mask].groupby('DayOfWeek')['Sales'].mean()
dw_mc = df[mask].groupby('DayOfWeek')['Customers'].mean()

dw_m.index = dw_m.index.map(day_names)
dw_mc.index = dw_mc.index.map(day_names)

# Plot the average number of sales on each day of the week
ax3.bar(dw_m.index, dw_m.values)
ax3.set_title('Average sales on a given day of the week')
ax3.set_ylabel('Customers')

# Plot the average number of customers on each day of the week
ax4.bar(dw_m.index, dw_m.values)
ax4.set_title('Average no.of customers on a given day of the week')
ax4.set_ylabel('Customers')

plt.suptitle("Average Sales and customer on given day of week for open stores")
plt.show()

# Average Sales on given month for open and "open and close" store

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))


dmonth_df = df.groupby('date_month')['Sales'].mean()
#dmonth_df_c = df.groupby('DayOfWeek')['Customers'].mean()
sns.barplot(x=dmonth_df.index, y=dmonth_df.values, ax=ax1)
ax1.set_title('Average sales on a given month for open store')
ax1.set_ylabel('Sales')


mask = df['Sales']>0
dmonth_m1 = df[mask].groupby('date_month')['Sales'].mean()
sns.barplot(x=dmonth_m1.index, y=dmonth_m1.values, ax=ax2)
ax2.set_title('Average sales on a given month for open and close store')
ax2.set_ylabel('Sales')
plt.show()

# save the artifact

#observations_text = 'Observations:\n1. On Monday, sales are high and Sunday is minimum for all stores.\n2. On Monday, customers are high and Sunday is minimum for all stores.\n3. The average sales and customers for open stores follow the same trend as all stores, but with slightly higher values.'

#task.get_logger().report_text(observations_text, 'observations', iteration=0)

# #observation for above graph 
# 1.On Monday sales high and sunday is minimum
# 2.On Monday customer is high and sunday is minimum

# In[30]:
train,test=train_test_split(df)


# In[31]:
# State Holiday
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

holiday_names={1 :"public" , 2 : "Easter",  3 : "Christmas", 0 : "None"}

#fig, (axis3,axis4) = plt.subplots(1,2,figsize=(15,4))
df["StateHoliday"].loc[df["StateHoliday"] == 0] = 0
sth_df = df.groupby('StateHoliday')['Date'].count()
sth_df.index=sth_df.index.map(holiday_names)
axis1.bar(sth_df.index, sth_df.values)
axis1.set_ylabel('Frequency')
axis1.set_title('Frequency of occurence of State holiday')

sth_df_s = df.groupby('StateHoliday')['Sales'].mean()
sth_df_s.index=sth_df_s.index.map(holiday_names)

axis2.bar(sth_df_s.index, sth_df_s.values)
axis2.set_title('Average sales on State holiday considering both Opended and closed stores')
axis2.set_ylabel('Sales')
plt.show()

fig, (axis3,axis4) = plt.subplots(1,2,figsize=(15,4))

mask = df["Sales"] > 0
df_copy = df.copy()
df_copy["StateHoliday"] = df[mask]["StateHoliday"].apply(lambda x:0 if x=="0" else 1)
sns.boxplot(x='StateHoliday', y='Sales', data=df_copy, ax=axis3)
axis3.set_title('Sales distribution on State holidays vs other days')

sth_df_s=df[mask].groupby('StateHoliday')['Sales'].mean()
sth_df.index=sth_df_s.index.map(holiday_names)
axis4.bar(sth_df_s.index, sth_df_s.values)
axis4.set_ylabel('Sales')
axis4.set_title('Average sales on State holiday considering only open stores')
plt.show()


# #Observations for above graph 1.frequency of occurancs of holiday a = public holiday, b = Easter holiday, c = Christmas, 0 = None there are less public holiday and easter holiday,and christmas holiday is very nrglisible StateHoliday 0 986159 1 20260 2 6690 3 4100
# 3% of record comes under holidays
# 2.Average sales on holidays(both for open and closed store) StateHoliday 0 5947.483893 1 290.735686 2 214.311510 3 168.733171
# if there is no holiday,then sales is high
# 4.Average sales on holidays(for open store) the order of rhe sales easter>christmas>public>no holiday

# In[32]:
# School Holiday

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,5))

#plt.subplots_adjust(wspace = 0.25)
sns.countplot(x='SchoolHoliday', data=df, ax = axis1)
axis1.set_title('Count of School holidays vs other days')

mask = mask = df["Sales"] > 0
sch_df = df[mask]

sns.boxplot(data = sch_df, x = 'SchoolHoliday', y = 'Sales', ax=axis2)
axis2.set_title('Boxplot of sales on School holidays vs other days')


sth_df_s1=df[mask].groupby('SchoolHoliday')['Sales'].mean()
sns.barplot(x=sth_df_s1.index, y=sth_df_s1.values, ax=axis3)
axis3.set_ylabel('Sales')
axis3.set_title('Average sales on school holiday considering only open stores')

fig.subplots_adjust(wspace=0.7)
plt.suptitle("PLots for school holidya",fontsize=12, y=1.05)
plt.show()


# In[33]:
(len(df[df["SchoolHoliday"]==0])/len(df))*100


# In[34]:
(len(df[df["SchoolHoliday"]==1])/len(df))*100


# #Observations for above graph

# 1.17% of data falls u under school holiday
# 2.bar plot value is almost same for both
# 3.When the public school are close(==0) the the sale is less then compared to when the schools are open(==1).

# In[35]:
# Store Type
#  
store_type1={0:"a",1:"b",2:"c",3:"d"}
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,5))
#fig, (axis3) = plt.subplots(1,figsize=(5,3))

df['Store_Type'] = df['StoreType'].replace(store_type1)
sns.countplot(x='Store_Type', data=df,  ax=axis1)
axis1.set_title('Number of stores of each type')

sns.boxplot(x='Store_Type', y='Sales', data=df[mask].replace(store_type1), ax=axis2)
axis2.set_title('Boxplot of sales for each store type')

#plt.suptitle("plot for storetype")
#plt.show()

#fig, (axis3) = plt.subplots(1,figsize=(5,3))
sth_df_s1 = df[mask].replace(store_type1).groupby('Store_Type')['Sales'].mean()

#sth_df_s1=df[mask].groupby('StoreType')['Sales'].mean()
sns.barplot(x=sth_df_s1.index, y=sth_df_s1.values, ax=axis3)
axis3.set_ylabel('Sales')
axis3.set_title('Average sales of each Storetype  only open stores')

fig.subplots_adjust(wspace=0.8)
plt.suptitle("plot for storetype",fontsize=12, y=1.05)
plt.show()


# #observation for above graph

# 1.a types stores are more,the order of stores are a>d>c>b 
# 2.the avg sales of stores b is higher compare to other store,the orders are b>a>c>d

# In[37]:
# Assortment  
fig, (axis1,axis2, axis3) = plt.subplots(1,3,figsize=(15,5))
#fig, (axis3) = plt.subplots(1,figsize=(5,3))
Assortment_type={0 :"basic(a)", 1 : "extra(b)",2 : "extended(c)"}
df["Assortment"] = df["Assortment"].map(Assortment_type)

sns.countplot(x='Assortment', data=df, ax=axis1)
axis1.set_title('No of stores having a particular Assortment type')

sns.boxplot(x='Assortment', y='Sales', data=df[mask], ax=axis2)
axis2.set_title('Boxplot of sales for stores selling an Assortment type')

sth_df_s1=df[mask].groupby('Assortment')['Sales'].mean()
sns.barplot(x=sth_df_s1.index, y=sth_df_s1.values, ax=axis3)
axis3.set_ylabel('Sales')
axis3.set_title('Average sales of each assortment type, only open stores')

fig.subplots_adjust(wspace=0.8)
fig.suptitle('Plots for Assortment Type', fontsize=12, y=1.05)


plt.show()


# #observation for above graph
 
# 1.the number of stores are more for assortment type a(basic),order is 0>2>1
# 2.but sales is more for assortment type b (extra),order is 1>0>2
# # Heatmap of Correlation in numerical vaules

# In[39]:
numerical=df[["CompetitionDistance",'CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2','Promo2SinceWeek',"Promo2SinceYear","Sales",'Open',"Promo","Customers","StateHoliday",'SchoolHoliday']]
plt.figure(figsize=(20, 10))
sns.heatmap(numerical.corr(), annot=True)
plt.show()


# # Plot of Promotion Data
# 

# In[40]:
#checking the data when the store has  no promotion
df_Open=df[(df.Promo2==0)]
df_Open.head()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

df_Open.hist('Promo2SinceWeek', ax=axes[0])
axes[0].set_title('Promo2SinceWeek')

df_Open.hist('Promo2SinceYear', ax=axes[1])
axes[1].set_title(' Promo2SinceYear')

fig.suptitle('Histogram for promo2sinceweek and promosince year when there is no promo2', fontsize=12, y=1.05)

plt.tight_layout()
plt.show()

#checking the data when the stores has promotion
df_close=df[(df.Promo2==1)]
df_close.head()

#plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

df_close.hist('Promo2SinceWeek', ax=axes[0])
axes[0].set_title('Promo2SinceWeek')

df_close.hist('Promo2SinceYear', ax=axes[1])
axes[1].set_title(' Promo2SinceYear')

fig.suptitle('Histogram for promo2sinceweek and promosince year when there is promo2', fontsize=12, y=1.05)

plt.tight_layout()
plt.show()


# In[41]:
df_Open.head()


# In[42]:
df_promo2_0 = df[df['Promo2'] == 0]


# In[43]:
df_promo2_0["Store"].unique()


# In[44]:
(len(df_promo2_0 )/len(df))*100


# In[45]:
df_promo2_1 = df[df['Promo2'] == 1]


# In[46]:
df_promo2_1["Store"].unique()


# In[47]:
(len(df_promo2_1 )/len(df))*100


# In[48]:
df_promo2_1["Promo2SinceWeek"].unique()


# In[49]:
len(df_promo2_1["Promo2SinceWeek"].unique())


# In[50]:
counts = df_promo2_1["Promo2SinceWeek"].value_counts()
print(counts)


# In[51]:
df_promo2_1["Promo2SinceYear"].unique()

# In[52]:
len(df_promo2_1["Promo2SinceYear"].unique())

# In[53]:
counts = df_promo2_1["Promo2SinceYear"].value_counts()
print(counts)


# #Observation from above

# 1.50% of stores not participating in promo2 and 50% of storeas are participating in promo2
# 2.the stores which are participating last 14  and 40 week of that year are high
# 3.the stores are participating from 1 to7 yeras range  but the stores which are  participating last 7 and 6 year that is since 2011 and 2013 are more

# In[55]:
#Checking when the stores are open nd sales are not equal to 0
df_open = df[~((df.Open ==0) | (df.Sales==0))]
df_open.head().T
#print(df_open.columns)


# In[56]:
#k=sns.factorplot(data = df_open, x='date_month', y='Sales',
 #             col ='Promo', hue='Promo2', row='date_year')


# In[57]:
fig, (axis1) = plt.subplots(1,figsize=(8,5))

sns.countplot(x='Open', data=df, ax=axis1)
axis1.set_title('Number of stores open')
plt.show()


# In[58]:
df_open1 = df[df["Open"]==1]
len(df_open1)/len(df)*100


# #Observation for store open
# 1.83% of stores are open and 17% of stores are close

# In[ ]:
print(df.head().T)

local_file_path = 'observations.txt'
artifact_object_path = os.path.join('Observationdata', 'observations.txt')

task.upload_artifact(local_file_path, artifact_object=artifact_object_path)



