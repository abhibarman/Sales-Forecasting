from clearml import Task, Dataset
import pandas as pd

import global_config
#from data import database


task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='get data',
    task_type='data_processing',
    reuse_last_task_id=False
)

# Create a ClearML dataset
dataset = Dataset.create(
    dataset_name='raw_sales_dataset',
    dataset_project=global_config.PROJECT_NAME
)
# Add the local files we downloaded earlier
dataset.add_files('Data/Sales.csv')
dataset.add_files('Data/store.csv')
dataset.add_files('Data/metadata.csv')

df = pd.read_csv("Data/Sales.csv")
df1 = pd.read_csv("Data/store.csv")
df_metadata=pd.read_csv('Data/metadata.csv')

#dataset.get_logger().report_table("sss" ,"ssss", url=urls[0])
# Let's add some cool graphs as statistics in the plots section!
dataset.get_logger().report_table(title='Sales data', series='head', table_plot=df.head())
dataset.get_logger().report_table(title='Store data', series='head', table_plot=df1.head())

#metadata 
dataset.set_metadata(df_metadata, metadata_name='Data information', ui_visible=True)

# Finalize and upload the data and labels of the dataset
dataset.finalize(auto_upload=True)

print(f"Created dataset with ID: {dataset.id}")
print(f" sales Data size: {len(df)}")
print(f"store Data size: {len(df1)}")
