package lab2oncesiorneksoru;

public abstract class Personel {
	private String adSoyad;
	private int id;
	private Daire ev;
	public Personel(int id, String adSoyad , Daire ev) {
		this.id=id;
		this.adSoyad=adSoyad;
		this.ev=ev;
	}
	public String getAdSoyad() {
		return adSoyad;
	}
	public int getId() {
		return id;
	}
	public Daire getEv() {
		return ev;
	}
	public abstract void kendiniTanit();
	public int getDaireKapiNo() {
		return ev.getKapiNo();
	}
}
