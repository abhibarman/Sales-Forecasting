#!/usr/bin/env python
# coding: utf-8

 # Import Libraries
 
import os.path
from pathlib import Path
import pandas as pd
from clearml import Dataset, Task
import global_config

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer 
     

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='Handle null values',
    task_type='data_processing',
    reuse_last_task_id=False
)

# Get the dataset
dataset = Dataset.get(
    dataset_project=global_config.PROJECT_NAME,
    dataset_name='raw_sales_dataset',
    alias='my_raw_dataset')

local_folder = dataset.get_local_copy()
print(f"Using dataset ID: {dataset.id}")

# Clean up the data a little bit
# # Sales Dataset
#traing data

df_train = pd.read_csv((Path(local_folder) / 'Sales.csv'))

df_train.head()

df_train.tail()

df_train.shape


#store data sets
df_store=pd.read_csv((Path(local_folder) / 'store.csv'))
df_store.head()


# In[6]:
df_store.tail()


# In[7]:
df_store.shape


# In[8]:
print(df_store[df_store.Promo2==1].shape)
print(df_store[df_store.Promo2==0].shape)


# In[9]:
(df_train.isnull().sum()/len(df_train))*100


# In[10]:
(df_store.isnull().sum()/len(df_store))*100


# In[11]:
df=df_store.merge(df_train,how='left')
print(df.isnull().sum())


# In[12]:
df.head()


# In[13]:
df.tail()


# In[14]:
df.shape


# In[15]:
sns.heatmap(df.isnull(),yticklabels=False,cbar=True,cmap='Accent')
plt.title("Fig 1:Missing Values in retail-sales Dataset")
plt.tight_layout()
plt.show()


# In[16]:
#histogram=df.hist(bins = 30, figsize=(20, 20), color = 'r')
for col in df.columns:
    print(col)
    data = df[col].dropna()
    data = data.values.tolist()
    plt.hist(data, color='r', bins=30,)
    plt.title("Histogram of {}".format(col))

    plt.xlabel(col)
    plt.ylabel('Freq')
    plt.show()

#task.get_logger().report_plot("Histogram of Data", iteration=0, figure=histogram)



# In[17]:
df.info()


# In[18]:
check_cols = [ 'StoreType','Assortment','CompetitionOpenSinceMonth',  'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear','DayOfWeek','Open', 'Promo', 'Promo2','SchoolHoliday']

for col in check_cols:
    print(col)
    print(sorted(df[col].unique()))
    print(df[col].value_counts())
    print()


# In[19]:
print(df['PromoInterval'].unique())
print(df["PromoInterval"].value_counts())


# In[20]:
print(df["StateHoliday"].unique())
print(df["StateHoliday"].value_counts())


# In[21]:
print(df['Store'].unique())
print(df["Store"].value_counts())


# # List of different DataTypes

# In[22]:
numerical=['Store','CompetitionDistance','Customers']

categorical_binary=['Open','Promo2','SchoolHoliday']


categorical_nominal=['StoreType','CompetitionOpenSinceMonth','Date','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','StateHoliday']

#column by col encoder
categorical_ordinal=['Assortment','DayOfWeek']
#label encoder

target=['Sales']


# # Fill NaN values
# # Filling up 'Promo2SinceWeek','Promo2SinceYear','PromoInterval' columns


# In[23]:
df['Promo2SinceWeek'].fillna('0',inplace=True)
df['Promo2SinceYear'].fillna('0',inplace=True)
df['PromoInterval'].fillna('0',inplace=True)

print(df.isnull().sum())


# # Filling up 'CompetitionDistance' column.


# In[24]:
imp_mean = SimpleImputer( strategy='mean') 
imp_mean.fit(df[['CompetitionDistance']])
df[['CompetitionDistance']] = imp_mean.transform(df[['CompetitionDistance']])


# # Filling up 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear'


# In[25]:
imp_mean = SimpleImputer( strategy='median') 
imp_mean.fit(df[['CompetitionOpenSinceMonth']])
df[['CompetitionOpenSinceMonth']] = imp_mean.transform(df[['CompetitionOpenSinceMonth']])


# In[26]:
imp_mean = SimpleImputer( strategy='median') 
imp_mean.fit(df[['CompetitionOpenSinceYear']])
df[['CompetitionOpenSinceYear']] = imp_mean.transform(df[['CompetitionOpenSinceYear']])


# In[27]:
print(df.isnull().sum())


# # Change of Datatype

# In[28]:
df['Assortment']=df['Assortment'].astype('category')
df['StoreType']=df['StoreType'].astype('category')
df['PromoInterval']=df['PromoInterval'].astype('category')
df['Promo2SinceWeek']=df['Promo2SinceWeek'].astype('category')
df['Promo2SinceYear']=df['Promo2SinceYear'].astype('category')
df['StateHoliday']=df['StateHoliday'].astype('category')
df['Date']=df['Date'].astype('datetime64')


# In[30]:
df.info()


# In[31]:
df.to_csv('MISSING_VALUE_HANDLED.csv',index=False)

new_dataset = Dataset.create(
    dataset_project=dataset.project,
    dataset_name='raw_sales_dataset',
    parent_datasets=[dataset]
)

new_dataset.add_files('MISSING_VALUE_HANDLED.csv')

new_dataset.get_logger().report_table(title='Missing value handled data', series='head', table_plot=df.head())
#new_dataset.get_logger().report_table(title='y data', series='head', table_plot=y.head())
new_dataset.finalize(auto_upload=True)

# Log to console which dataset ID was created
print(f"Created preprocessed dataset with ID: {new_dataset.id}")

# In[ ]:




