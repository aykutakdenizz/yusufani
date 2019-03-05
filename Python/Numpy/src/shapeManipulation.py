import numpy as np

a = np.floor(10*np.random.random((3,4)))

print(a)
print (a.shape) # 3 e 4 lük bir dizi
print (a.ravel()) # matrisi düz bir dizi yap
a=a.ravel()
print(a.reshape(2,6)) # tekrar matrise cevirdik
a=a.reshape(2,6)
print(a.T)
print(a.reshape(2,-1)) # 2 eşit satır haline getir
