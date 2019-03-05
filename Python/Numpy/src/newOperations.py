import numpy as np

a = np.array([20,30,40,50])
print(a)
b = np.arange(4)
print(b)
c= a-b 
print(c)
d = b**3
print(d)
e = 10* np.sin(a)
print(e<7)
print(a*b) # element wise çarpım
print(a@b) # Matris çarpımı
print (a.dot(b)) # Bu da matris çarpımı

f = np.ones((2,4))
g = np.zeros((2,4))
h = np.random.random((2,4))
i = np.sum(b)
print(b.sum())
k =np.max(h)
l = np.sqrt(b)
