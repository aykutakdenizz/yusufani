public class Test {
    public static void main(String[] args) throws YetersizPuanException {
        Musteri m[] = new Musteri[4];
        m[0]=new Standart(1, "Ali Ertas", 100, 10);
        m[1]=new Standart(1, "Ayse Genc", 100, 15);
        m[2]=new Elite(1, "Ahmet Yılmaz" , 100, 20);
        m[3]=new Elite(1, "Fatma Ozturk", 100, 25);
        BankaYonetim b=new BankaYonetim();
        try{
            b.musteriEkle(m[0]);
            b.musteriEkle(m[1]);
            b.musteriEkle(m[2]);
            b.musteriEkle(m[3]); // Tek Tek ekleme
        }
        catch(MaxMusteriSayisiException e) {
            e.printStackTrace();
        }
        try{
            b.musteriEkle(m); // Collections.addAll(list, values); banka içinde böyle bir kod kullanılabilir//Burada da MaxMusteriSayısıException Oluşabilir
        }catch (MaxMusteriSayisiException e){
            e.printStackTrace();
        }


        m[0].alisverisYap(100); // 100 TL değerinde alışveriş yapılıyor. Hem puan güncelleme yapılmalı hem de alışveriş taahhüt değeri değeri 1 azaltılmalıdır.
        m[1].setFaydalandıgıAyKadarAldigiBonusPuanlar(300);
        m[1].setTaaahutluAlısverisSayisi(9);
        m[1].setIslemSayisi(5);
        m[1].setPuan(400);
        try{
            b.musteriSil(m[1]);
        }catch(TaahhutSuresiDolmadiException e){
            e.printStackTrace();
        }
        catch(YetersizPuanException y){
           y.printStackTrace();
        }
        b.listele();
    }
}
