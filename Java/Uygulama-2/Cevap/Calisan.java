package lab2;

public class Calisan {
	
protected String ad;
protected  int yas;
protected int  kidem;
protected int maas;

public Calisan()
{}
public Calisan(String ad, int yas)
{
    this.ad=ad;
    this.yas=yas;
}

public void setMaas(int maas) {this.maas=maas;}

public void setKidem (int kidem) {this.kidem=kidem;}  

public int  getMaas () {return this.maas;}
public int  getKidem () {return this.kidem;}

public void kendiniTanit(){
   System.out.println("Çalışan");
}
}
