public abstract class Urun {
	private int id;
	private String urunAdi;
	private double fiyat;
	public Urun(int id, String urunAdi, double fiyat) {
		this.id = id;
		this.urunAdi = urunAdi;
		this.fiyat = fiyat;
	}
	public double getFiyat() {
		return fiyat;
	}
	public void setFiyat(double fiyat) {
		this.fiyat = fiyat;
	}
	public abstract void fiyatGuncelle ();
	public String toString() {
		return urunAdi+" "+Integer.toString(id)+" "+Double.toString(fiyat);
	}
}
