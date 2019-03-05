import numpy as np
a =np.floor(10* np.random.random((2,3)))
b =np.floor(10* np.random.random((2,3)))
print(a)
print(b,"\n Dikey olarak Birlesmis hali:")
c = np.vstack((a,b))
print (c) 
c = np.hstack((a,b))
print("yatay olarak Birlesmis hali:",c)