package lab2kendimdeneme;

public class ArastirmaGorevlisi extends Calisan {
	private String durum;
	public  void doktoraDurumGoster(String durum)
	{
	     this.durum=durum;
	}
	public ArastirmaGorevlisi(String ad,int yas, int kidem, int maas ) {
		super(ad,yas);
		this.kidem=kidem;
		this.maas=maas;
	}
	public void kendiniTanıt() {
		  System.out.println("Adim: " + this.ad + " Yasim : "+ Integer.toString(this.yas)+ " Kidemim " + this.getKidem() 
          + " yil,  Maasim: " + this.getMaas() + " Durumum: " + this.durum );    
	}
	
}
