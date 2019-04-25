public class Person {
    private int id;
    private String isim,soyisim,sifre;
    private char cinsiyet;// ERKEK -> 0 KADIN ->1
    public Person(String isim, String soyisim, String sifre,char cinsiyet) {
        this.isim = isim;
        this.soyisim = soyisim;
        this.sifre = sifre;
        this.cinsiyet=cinsiyet;
    }

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
    public char getCinsiyet() {
        return cinsiyet;
    }

    public void setCinsiyet(char cinsiyet) {
        this.cinsiyet = cinsiyet;
        if(cinsiyet-48>1 || cinsiyet-48<0){
            System.out.println("Cinsiyet Erkek veya Kadin olabilir.");
            this.cinsiyet=0;
        }
    }
}
