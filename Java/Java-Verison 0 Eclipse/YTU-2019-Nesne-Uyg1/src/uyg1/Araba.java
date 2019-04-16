package uyg1;

public class Araba {
	private String plaka,marka;
	private int model;
	public Araba() {
	
	}
	public Araba(String plaka, String marka, int model) {
		this.plaka = plaka;
		this.marka = marka;
		this.model = model;
	}
	public String getPlaka() {
		return plaka;
	}
	public void setPlaka(String plaka) {
		this.plaka = plaka;
	}
	public String getMarka() {
		return marka;
	}
	public void setMarka(String marka) {
		this.marka = marka;
	}
	public int getModel() {
		return model;
	}
	public void setModel(int model) {
		this.model = model;
	}
	public void arabaBilgileriniGöster(){
		System.out.println("Araba Markasi:"+this.getMarka()+" Modeli:"+this.getModel()+" Plaka"+this.getPlaka());
	}
	
}
