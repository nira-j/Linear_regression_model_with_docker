import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

df=pd.read_csv('DataSalary.csv')
x=df['YearsExperience']
x=np.array(x)
x=x.reshape(-1,1)

y=df['Salary']

y=np.array(y)
y=y.reshape(-1,1)


model=LinearRegression()
model.fit(x,y)

joblib.dump(model,'salary_predict.pk1')
