#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[21]:
import os.path
from pathlib import Path
from clearml import Dataset, Task
import global_config

import pandas as pd
import numpy as np
from sklearn import preprocessing

# In[22]:

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='Feature_engineering',
    task_type='data_processing',
    reuse_last_task_id=False
)

dataset1 = Dataset.get(
    dataset_project=global_config.PROJECT_NAME,
    dataset_name='raw_sales_dataset',
    alias='my_preprocess_dataset1'
)
local_folder = dataset1.get_local_copy()
print(f"Using dataset ID: {dataset1.id}")

df = pd.read_csv((Path(local_folder) / 'MISSING_VALUE_HANDLED.csv'))
df.info()


# In[23]:
df['Assortment']=df['Assortment'].astype('category')
df['StoreType']=df['StoreType'].astype('category')
df['PromoInterval']=df['PromoInterval'].astype('category')
df['Promo2SinceWeek']=df['Promo2SinceWeek'].astype('category')
df['Promo2SinceYear']=df['Promo2SinceYear'].astype('category')
df['StateHoliday']=df['StateHoliday'].astype('category')
df['Date']=df['Date'].astype('datetime64')


# In[24]:
df.info()


# # List of Different DataType

# In[25]:
numerical=['Store','CompetitionDistance','Customers']

categorical_binary=['Open','Promo2','SchoolHoliday']


categorical_nominal=['StoreType','CompetitionOpenSinceMonth','Date','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','StateHoliday']


categorical_ordinal=['Assortment','DayOfWeek']


target=['Sales']


# # Label Encoding


# In[26]:
label_encoder = preprocessing.LabelEncoder() 
df['Assortment']= label_encoder.fit_transform(df['Assortment']) 
df['Assortment'].unique()


# In[27]:
df['DayOfWeek']= label_encoder.fit_transform(df['DayOfWeek']) 
df['DayOfWeek'].unique()


# # One-Hot Encoding

# # Encoding for Binary data

# In[28]:
print(df['Open'].unique())
print(df['Promo2'].unique())
print(df['SchoolHoliday'].unique())


# In[29]:
encoded1=pd.get_dummies(df[['Open','Promo2','SchoolHoliday']],columns=['Open','Promo2','SchoolHoliday'],drop_first=False)
encoded1


# In[30]:
df=pd.concat([df,encoded1],axis=1)


# In[31]:
df.head()


# # Encoding for nominal data

# In[32]:
print(df['StoreType'].unique())
print(df['PromoInterval'].unique())
print(df['StateHoliday'].unique())


# In[33]:
PromoInterval_mapping={'0':'0','Jan,Apr,Jul,Oct':'Jan_Apr_Jul_Oct','Feb,May,Aug,Nov':'Feb_May_Aug_Nov','Mar,Jun,Sept,Dec':'Mar_Jun_Sept_Dec'}

df['PromoInterval'] = df['PromoInterval'].map(PromoInterval_mapping)


# In[34]:
print(df['PromoInterval'].unique())


# In[35]:
encoded2=pd.get_dummies(df[['StoreType','PromoInterval','StateHoliday']],columns=['StoreType','PromoInterval','StateHoliday'])
encoded2


# In[36]:
df=pd.concat([df,encoded2],axis=1)
df.head()


# In[37]:
print(df['StoreType'].unique())
print(df['PromoInterval'].unique())
print(df['StateHoliday'].unique())


# In[38]:
StoreType_mapping={'a':0,'b':1,'c':2,'d':3}
PromoInterval_mapping={'0':0,'Jan_Apr_Jul_Oct':1,'Feb_May_Aug_Nov':2,'Mar_Jun_Sept_Dec':3}
StateHoliday_mapping={'0':0,'a':1,'b':2,'c':3}

df['StoreType']     = df['StoreType'].map(StoreType_mapping)
df['PromoInterval'] = df['PromoInterval'].map(PromoInterval_mapping)
df['StateHoliday']     = df['StateHoliday'].map(StateHoliday_mapping)


# # Convert/Manage Date column
# 

# In[39]:
def extend_date_feature(my_df):
    my_df['Date']=my_df['Date'].astype('datetime64[ns]')
    my_df.loc[:,'date_year']=my_df['Date'].apply(lambda x: x.year)
    my_df.loc[:,'date_weekofyear']=my_df['Date'].apply(lambda x: x.weekofyear )
    my_df.loc[:,'date_month']=my_df['Date'].apply(lambda x: x.month )
    my_df.loc[:,'date_dayofweek']=my_df['Date'].apply(lambda x: x.dayofweek )
    my_df.loc[:,'date_day']=my_df['Date'].apply(lambda x: x.day )
    #my_df=my_df.drop('Date',axis=1)
    return my_df

f=extend_date_feature(df)


# In[40]:
df.info()


# In[41]:
df.to_csv('feature_engneering.csv',index=False)


# In[ ]:
new_dataset1 = Dataset.create(
    dataset_project=dataset1.project,
    dataset_name='raw_sales_dataset',
    parent_datasets=[dataset1]
)
new_dataset1.add_files('feature_engneering.csv')
new_dataset1.get_logger().report_table(title='feature engineering data', series='head', table_plot=df.head())
#new_dataset.get_logger().report_table(title='y data', series='head', table_plot=y.head())
new_dataset1.finalize(auto_upload=True)

# Log to console which dataset ID was created
print(f"Created preprocessed dataset with ID: {new_dataset1.id}")




