
public class Musteri extends Person{
    private String saatler,hareketler1,hareketler2,hareketler3;
    private double VKI;
    private double kasOrani;
    public Musteri(int id, String isim, String soyisim, String sifre, String saatler, String hareketler1, String hareketler2, String hareketler3, double VKI, double kasOrani) {
        super(id, isim, soyisim, sifre);
        this.saatler = saatler;
        this.hareketler1 = hareketler1;
        this.hareketler2 = hareketler2;
        this.hareketler3 = hareketler3;
        this.VKI = VKI;
        this.kasOrani = kasOrani;
    }
    public String getSaatler() {
        return saatler;
    }

    public void setSaatler(String saatler) {
        this.saatler = saatler;
    }

    public String getHareketler1() {
        return hareketler1;
    }

    public void setHareketler1(String hareketler1) {
        this.hareketler1 = hareketler1;
    }

    public String getHareketler2() {
        return hareketler2;
    }

    public void setHareketler2(String hareketler2) {
        this.hareketler2 = hareketler2;
    }

    public String getHareketler3() {
        return hareketler3;
    }

    public void setHareketler3(String hareketler3) {
        this.hareketler3 = hareketler3;
    }

    public double getVKI() {
        return VKI;
    }

    public void setVKI(double VKI) {
        this.VKI = VKI;
    }

    public double getKasOrani() {
        return kasOrani;
    }

    public void setKasOrani(double kasOrani) {
        this.kasOrani = kasOrani;
    }

    public void dersProgramiOlustur(VeriTabani x){
        System.out.println("Ders Programi Olusturucuya Hosgeldiniz");
        StringBuilder mesaj = new StringBuilder("Vucut kitle indexine ve kas oranınınıza göre size uygun olan program ");
        if ( this.VKI < 20 && this.kasOrani > 36   ){
            mesaj.append("bolgesel antrenman tipidir.");
            System.out.println(mesaj);
            System.out.println("Sizin için uygun antrenman programı su sekildedir:buraya indislere göre degerler gelecek");
            this.setHareketler1("1;2;3;4;6;");
            this.setHareketler2("7;8;9;");
            this.setHareketler3("10;11;12");
        }
        else {
            mesaj.append("tum vucut antrenman tipidir");
            this.setHareketler1("1;2;3;4;6;");
            this.setHareketler2("1;2;3;4;6;");
            this.setHareketler3("1;2;3;4;6;");
        }
        x.musteriDersProgramıGuncelle(this);
    }
}
