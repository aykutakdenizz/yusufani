package lab2kendimdeneme;

import java.util.ArrayList;

public class Bolum {
	private String bolumAdi,adres;
	private int kurulus,toplamMaas=0;
	private ArrayList<Calisan> arr = new ArrayList<Calisan>();
	public Bolum (String bolumAdi, String adres, int kurulus) {
		this.adres=adres;
		this.bolumAdi=bolumAdi;
		this.kurulus=kurulus;
	}
	public void calisanEkle(Calisan obj) {
		arr.add(obj);
	}
	public void calisanToplamMaas() {
		for(Calisan x : arr)
			toplamMaas+=x.getMaas();
		System.out.println("Toplam maas"+ toplamMaas);
	}
	
}
