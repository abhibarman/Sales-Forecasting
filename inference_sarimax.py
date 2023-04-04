from clearml import Task
import pickle 
import matplotlib.pyplot as plt

#get the task
tsk = Task.init('Retail_Sales_Forecasting10','Sarimax Inference','inference') 
task = Task.get_task('7b30f2f252e04d15be209f68d23f4d18') 
args = {'steps' : 42} 
tsk.connect(args) 

#load the tokenizer 
model_path = task.artifacts['sarimax_model'].get_local_copy() 
with open(model_path, 'rb') as handle:
     model = pickle.load(handle) 
     res = model.forecast(steps=args['steps']) 
     print(res) 
     res.plot()
     plt.title("Forcasted Sale")
     plt.show()
tsk.close()