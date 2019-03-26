package lab2kendimdeneme;

public class OgretimUyesi  extends Calisan{
	private String unvan = "";
	public OgretimUyesi (String ad , int yas , int kidem,int maas) {
	    super(ad,yas);
	    this.setKidem(kidem);
	    this.setMaas(maas);   
	}
	public  void setUnvan(String unvan)
	{
	     this.unvan=unvan;
	}
	public String getUnvan()
	{
	    return this.unvan;
	} 
	public void kendiniTanıt() {
		  System.out.println("Adim: " + this.ad + " Yasim : "+ Integer.toString(this.yas)+ " Kidemim " + this.getKidem() 
        + " yil,  Maasim: " + this.getMaas() + " Durumum: " + this.getUnvan() );    
	}

}
