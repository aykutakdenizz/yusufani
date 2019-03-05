import pandas as pds
data = [10,20,30,40,50]
df = pds.DataFrame(data)
print(df)
data2 = [["Yusuf","Anı",20],["fazlı","Anı",21],["Ahmet","Anı",22]   ]
df2 = pds.DataFrame(data2,columns = ["isim","Soyad","yas"],index = [1,2,3])
print(df2)
data3 ={"İsim":["Yusuf","Ahmet","Fazli"],
        "Soyad":["anı","Aydı","BOz"],
        "Yas":[21,22,23]
        }
df3 = pds.DataFrame(data3,columns = ["İsim","Soyad","Sehir"],index = [1,2,3])
print(df3) 
#del df3["Soyad"] # soyadı silmek için
#df3.pop("Soyad") # yukarıdaki ile tamamen aynı 
print(df3)
print(df3.loc[2]) # 2. indexdeki bilgileri başlıklarıyla beraber yazddırma işlemi
print(df3.iloc[1]) #  index sırasıyla 1. indexdeki bilgileri başlıklarıyla beraber yazddırma işlemi
df4 = df3.append(df2)
print(df4)
print(df4.head(6)) # en üstteki 6 elamanı görmek 