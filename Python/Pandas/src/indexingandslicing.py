import pandas as pd
notlar = pd.read_csv("grades.csv")
print(notlar)
notlar.columns = ["Soyisim","İsim","ssn","test1","test2","test3","test4","final","sonuc"]
print (notlar)
print(notlar.loc[:,"İsim"])
print(notlar.loc[:5,"Soyisim"])
#print(notlar.loc[:5,["isim","Final"]])