public class Personel extends Person {
    public Personel( String isim, String soyisim, String sifre, char cinsiyet) {
        super( isim, soyisim, sifre, cinsiyet);
    }
    public double ortalamaMemnuniyetDuzeyiGoruntule(VeriTabani x){
        return x.ortalamaMemnuniyetDuzeyi();
    }

    @Override
    public String toString() {
        String tmp= getId()+"\t"+getIsim()+" "+getSoyisim()+"\t";
        if (getCinsiyet() == 0) tmp+="Erkek";
        else tmp+="Kadin";
        return tmp;
    }

}
