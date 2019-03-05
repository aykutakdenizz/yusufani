# -*- coding: utf-8 -*-

ogrenciler = ["Engin","Yusuf","Salih","Ali","Ayşe"]
fileToAppend =open("ogrenciler.txt","a")

for ogrenci in ogrenciler:
    fileToAppend.write(ogrenci+"\n")
fileToAppend.close()