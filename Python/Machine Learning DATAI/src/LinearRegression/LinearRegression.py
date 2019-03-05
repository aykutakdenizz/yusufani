import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv("linear-regression-dataset.csv",sep=";") # Aralarinda ; karakteri var 

# Plotting data
plt.scatter(df.deneyim,df.maas) # 2 farkli niteligin oldugunu belirledik
plt.xlabel("Deneyim")
plt.ylabel("Maas")
plt.show()

#%% sklearn

from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()
x = df.deneyim.values.reshape(-1,1) # 1 sutuna sigdir diyoruz -1 ise row saysini bilmedigimizden yazilir
y= df.maas.values.reshape(-1,1)
linear_reg.fit(x,y)   # tabloya yerlestirildi

b0= linear_reg.predict([[0]]) # 0 da kestigi degeri bulam
print("b0 =", b0)

b0 = linear_reg.intercept_ # kesistigi ilk nokta digeri ile ayni olur

b1=linear_reg.coef_
print("b1:",b1) # egimi bulduk

#maas = 1663 + 1138*deneyim
maas_yeni = b0+b1*10 # 10 yil deneyimi olan biri icin �nerilen maas
print("10 yillik deneyimi olan birisi icin  onerilen maas ",maas_yeni)

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) # deneyim 

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array) # maas degerini daha önce elde ettigimiz modele uyguladik
plt.plot(array,y_head,color = "red")
linear_reg.predict([[100]])
