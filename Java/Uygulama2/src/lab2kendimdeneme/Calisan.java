package lab2kendimdeneme;

public class Calisan {
	protected String ad;
	protected int yas,kidem,maas;
	protected Calisan (String ad,int yas) {
		this.ad=ad;
		this.yas=yas;
	}
	public void setMaas(int maas) {
		this.maas=maas;
	}
	public void setKidem(int kıdem) {
		this.kidem=kıdem;
	}
	public int getMaas() {
		return this.maas;
	}
	public int getKidem() {
		return this.kidem;
	}
	public void kendiniTanıt() {
		System.out.println("Calisan");
	}
	
}
