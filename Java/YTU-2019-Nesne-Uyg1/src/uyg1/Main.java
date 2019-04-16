package uyg1;

import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		System.out.println("Lutfen kapasiteyi giriniz:");
		Araba araba1 = new Araba();
		araba1.setMarka("Marka1");
		araba1.setModel(2015);
		araba1.setPlaka("1234");
		Araba araba2 = new Araba("5678","Marka2",2016);
		Araba araba3 = new Araba("9999","Marka3",2017);
		int kapasite=in.nextInt();
		Otopark otopark = new Otopark("Garage",kapasite);
		otopark.arabaEkle(araba1);
		otopark.arabaEkle(araba2);
		otopark.arabaEkle(araba3);
		otopark.otoparkTanit();
		otopark.arabaCikar("1234");
		otopark.otoparkTanit();
		/*
		otopark.kapasiteArttir();
		otopark.otoparkTanit();*/
		System.out.println(otopark.getKapasite());
		in.close();
	}

}
