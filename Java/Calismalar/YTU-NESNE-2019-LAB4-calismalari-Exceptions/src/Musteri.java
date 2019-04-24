public abstract class Musteri {
    private int id,puan,taaahutluAlısverisSayisi;
    private String adSoyad;
    private int islemSayisi;
    private int faydalandıgıAyKadarAldigiBonusPuanlar;

    public int getFaydalandıgıAyKadarAldigiBonusPuanlar() {
        return faydalandıgıAyKadarAldigiBonusPuanlar;
    }

    public void setFaydalandıgıAyKadarAldigiBonusPuanlar(int faydalandıgıAyKadarAldigiBonusPuanlar) {
        this.faydalandıgıAyKadarAldigiBonusPuanlar = faydalandıgıAyKadarAldigiBonusPuanlar;
    }

    public Musteri(int id, String adSoyad, int puan, int taaahutluAlısverisSayisi ) {
        this.id = id;
        this.puan = puan;
        this.taaahutluAlısverisSayisi = taaahutluAlısverisSayisi;
        this.adSoyad = adSoyad;
        islemSayisi=0;
    }

    public int getIslemSayisi() {
        return islemSayisi;
    }

    public int getPuan() {
        return puan;
    }

    public int getId() {
        return id;
    }

    public String getAdSoyad() {
        return adSoyad;
    }

    public int getTaaahutluAlısverisSayisi() {
        return taaahutluAlısverisSayisi;
    }
    public abstract  void alisverisYap( int x );

    public void setPuan(int puan) {
        this.puan = puan;
    }

    public void setTaaahutluAlısverisSayisi(int taaahutluAlısverisSayisi) {
        this.taaahutluAlısverisSayisi = taaahutluAlısverisSayisi;
    }

    public void setIslemSayisi(int islemSayisi) {
        this.islemSayisi = islemSayisi;
    }
}
