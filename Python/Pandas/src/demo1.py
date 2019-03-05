import pandas as pd
notlar = pd.read_csv("grades.csv")
print(notlar)
notlar.columns = ["Soyisim","İsim","ssn","test1","test2","test3","test4","final","sonuc"]
print (notlar)