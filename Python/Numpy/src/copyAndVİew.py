import numpy as np

a = np.arange(10)
print (a)
b = a
print (b)
b [0]=100
print (a)
print (b)
c = a.copy()
c[0]=25
print(c)
print(a)
d = a.view()
print("****")
print(a)
print(d)
d[0]=250
print(a)
print(d)
d.shape = 2,5
print(a)
print(d)
a[0]=123
print(a)
print(d)