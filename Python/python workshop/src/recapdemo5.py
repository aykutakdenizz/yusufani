import sys

liste = [7,'Engin',0,3,"6"]
for x in liste:
    try:
        print("Sayi"+str(x))
        sonuc = 1/int(x)
        print("Sonuc",sonuc)
    except:
        print(str(x)+"hesaplanamadi")
        print("Sistem diyor ki:",sys.exc_info()[0])
        