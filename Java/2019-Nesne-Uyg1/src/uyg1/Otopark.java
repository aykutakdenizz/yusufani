package uyg1;

import java.util.LinkedList;
import java.util.Scanner;

public class Otopark {
	private String otoparkAd;
	private int kapasite;
	LinkedList<Araba> arabalar = new LinkedList<Araba>();
	public Otopark(String otoparkAd, int kapasite) {
		this.otoparkAd = otoparkAd;
		this.kapasite = kapasite;
	}
	public boolean bosMu() {
		return arabalar.isEmpty();
		/*if(arabalar.size()== 0) return true;
		else return false;*/
	}
	public void arabaEkle(Araba araba) {
		if (arabalar.size() == this.kapasite) System.out.println("Kapasite yetersiz");
		else {
			this.arabalar.add(araba);
		}
	}
	public void arabaCikar(String plaka) {
		if (arabalar.size() <= 0 ) {
			System.out.println("Otoparkta hic araba yok");
		}
		else {
			for(Araba x : arabalar) {
				if(x.getPlaka().compareTo(plaka)== 0 ) {
					arabalar.remove(x);
					return;
				}
			}
		}
		
	}
	public void kapasiteArttir() {
		System.out.println("Lutfen arttirmak istediginiz kapasite miktarini giriniz:");
		Scanner in = new Scanner(System.in);
		int tmp = in.nextInt();
		this.kapasite+=tmp;
		in.close();
	}
	public int getKapasite() {
		return kapasite;
	}
	public void otoparkTanit() {
		System.out.println("Otopark Adi:"+this.otoparkAd+"Kapasitesi:"+this.kapasite);
		System.out.println("Arabalar\nMarka Model Plaka");
	
		for(Araba x : arabalar)
			System.out.println(x.getMarka()+" "+x.getModel()+" "+x.getPlaka()+"\n");
	}
	
}
