import numpy as np

havadurumu = [[12,21,31],[6,17,18],[11,12,13]]
#print(havadurumu)

a = np.arange(15).reshape(3,5)
print(a)
print(type(a))
print("Dimension Count =",a.ndim)
b=np.arange(10)
print(b.shape)# 1 boyutlu 10 elemanlı 
print(b.ndim)# boyutu verir