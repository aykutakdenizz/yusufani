package lab2orneksoru;
public class ArastirmaGorevlisi extends Calisan{	
private String durum;
public  void doktoraDurumGoster(String durum)
{
     this.durum=durum;
}
public ArastirmaGorevlisi(String ad, int yas, int kidem, int maas)
{
    super(ad,yas);
    this.setKidem(kidem);
    this.setMaas(maas);   
}
public void kendiniTanit()
{
    System.out.println("Adım: " + this.ad + " Yaşım : "+ Integer.toString(this.yas)+ " Kıdemim " + this.getKidem() 
            + " yıl,  Maaşım: " + this.getMaas() + " Durumum: " + this.durum );    
} 

}
