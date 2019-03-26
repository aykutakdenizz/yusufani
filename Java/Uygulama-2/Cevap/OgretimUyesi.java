package lab2;
public class OgretimUyesi extends Calisan {	
private String unvan="";

public  void setUnvan(String unvan)
{
     this.unvan=unvan;
}
public String getUnvan()
{
    return this.unvan;
} 
public OgretimUyesi(String ad, int yas, int kidem, int maas)
{
    super(ad,yas);
    this.setKidem(kidem);
    this.setMaas(maas);   
}
public void kendiniTanit()
{
    System.out.println("Adım: " + this.ad + " Yaşım : "+ Integer.toString(this.yas)+ " Kıdemim " 
            + this.getKidem() + " yıl,  Maaşım: " + this.getMaas() + "tl  Unvan : " + this.getUnvan() );   
} 
}
