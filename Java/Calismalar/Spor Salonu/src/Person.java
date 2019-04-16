public class Person {
    private int id;
    private String isim,soyisim,sifre;

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getIsim() {
        return isim;
    }

    public void setIsim(String isim) {
        this.isim = isim;
    }

    public String getSoyisim() {
        return soyisim;
    }

    public void setSoyisim(String soyisim) {
        this.soyisim = soyisim;
    }

    public String getSifre() {
        return sifre;
    }

    public void setSifre(String sifre) {
        this.sifre = sifre;
    }

    public Person(int id, String isim, String soyisim, String sifre) {
        this.id = id;
        this.isim = isim;
        this.soyisim = soyisim;
        this.sifre = sifre;
    }
}
