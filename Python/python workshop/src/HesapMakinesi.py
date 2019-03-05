selected = int(input("Operasyon ?  \n1:Topla 2:Çıkar 3:Bol 4:Çarp"))
sayi1=int(input("Sayi1'i giriniz:"))
sayi2=int(input("Sayi2'yi giriniz:"))
if selected == 1 :
    print(sayi1+sayi2)
if selected == 2 :
    print(sayi1-sayi2)
if selected == 3 :
    if (sayi2!=0):
        print(sayi1/sayi2)
    else : print("Sayi 0'a bölünemez")
if selected == 4 :
    print(sayi1+sayi2)
