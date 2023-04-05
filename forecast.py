import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task
import pickle
from numerize import numerize

def get_dataframe(num_rows):

    task = Task.get_task('d3bf17a06e49451bb82bf3352a271eea')
    args = {'steps' : num_rows}
    model_path  = task.artifacts['sarimax_model'].get_local_copy()

    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)
    res = model.forecast(steps=args['steps'])
    df = pd.DataFrame(res)
    df.reset_index(inplace=True)
    df.columns = ['timestamp','predicted_mean' ]
    return df


st.title('Retail Store Sales Forecast')

# Get user input
num_rows = st.number_input('Forcast for next:', min_value=1, max_value=1000, value=28)

# Generate the DataFrame
df = get_dataframe(num_rows)

dates = np.array(df["timestamp"].tolist())
sales = np.array(df["predicted_mean"].tolist())

# Calculate the minimas and the maximas
minimas = (np.diff(np.sign(np.diff(sales))) > 0).nonzero()[0] + 1 
maximas = (np.diff(np.sign(np.diff(sales))) < 0).nonzero()[0] + 1

# Plot the entire data first
fig, ax = plt.subplots()
plt.plot(dates, sales)
# Then mark the maximas and the minimas
for minima in minimas:
    plt.plot(df.iloc[minima]["timestamp"], df.iloc[minima]["predicted_mean"], marker="o")
    y_label = numerize.numerize(df.iloc[minima]['predicted_mean'])
    ax.text(df.iloc[minima]["timestamp"], df.iloc[minima]["predicted_mean"], f"{y_label}", size=12)
    
for maxima in maximas:
    plt.plot(df.iloc[maxima]["timestamp"], df.iloc[maxima]["predicted_mean"], marker="o")
    y_label = numerize.numerize(df.iloc[maxima]['predicted_mean'])
    ax.text(df.iloc[maxima]["timestamp"], df.iloc[maxima]["predicted_mean"], f"{y_label}", size=12)

#plt.legend()
ax.set_title(f'Sales for next {num_rows} Days')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Sales')
plt.xticks(rotation=45)
plt.show()

# Display the plot
st.pyplot(fig)
