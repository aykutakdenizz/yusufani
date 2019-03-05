import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("decision-tree-regression-dataset.csv",sep=";")

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#Decision tree regression

from sklearn.tree import DecisionTreeRegressor 
tree_reg  = DecisionTreeRegressor
tree_reg.fit(x,y)

y_head= tree_reg.predict(5.5)

plt.scatter(x, y, color ="red")
plt.plot(x,y_head,color="green")
plt.xlabel("Tribun level")
plt.ylabel("Ucret")
plt.show()
