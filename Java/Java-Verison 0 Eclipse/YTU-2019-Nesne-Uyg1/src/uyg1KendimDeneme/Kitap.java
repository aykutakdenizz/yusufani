package uyg1KendimDeneme;

public class Kitap {
	private String kitapAd,yazar,ISBN;
	private int rafNo,sira;

	public Kitap(String kitapAd, String yazar, String ISBN, int rafNo, int sira) {
		this.kitapAd = kitapAd;
		this.yazar = yazar;
		this.ISBN = ISBN;
		this.rafNo = rafNo;
		this.sira = sira;
	}

	public String getKitapAd() {
		return kitapAd;
	}

	public String getYazar() {
		return yazar;
	}

	public String getISBN() {
		return ISBN;
	}

	public int getRafNo() {
		return rafNo;
	}

	public int getSira() {
		return sira;
	}
	
}
