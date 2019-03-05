import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
data = pd.read_csv("hw_25000.csv")
boy = data.Height.values.reshape(-1,1)
kilo = data.Weight.values.reshape(-1,1)
regression = LinearRegression()
regression.fit(boy,kilo)
print (regression.predict(np.array([[70]])))
print(data.columns)


plt.scatter(data.Height,data.Weight)
x= np.arange(min(data.Height),max(data.Height)).reshape(-1,1)
plt.plot(x,regression.predict(x),color="red")
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Simple Linear Regression Model")
plt.show()
