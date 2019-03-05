import pandas as pd
import numpy as np
s = pd.Series()
data = np.array(["Yusuf","Ahmet","Salih"])
s= pd.Series(data) # indexleme yapıyor
print (s) 
s= pd.Series(data,index=[1,2,3]) # indexleme yapıyor
print(s)
data2= {"Matematik ":10,"fizik":20,"Beden":90}
s2 = pd.Series(data2)
print(s2)
data2= {"Matematik":10,"fizik":20,"Beden":90}
s2 = pd.Series(data2,index = ["fizik","Matematik","Beden"]) # Fizikle mat yer değişti
print(s2)
s3 = pd.Series(5,index=[1,2,3])
print(s3)