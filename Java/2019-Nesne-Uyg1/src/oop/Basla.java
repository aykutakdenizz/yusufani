package oop;

import java.util.Scanner;

public class Basla {

	public static void main(String[] args) {		
		int rafsayisi = 5;
		
		Kitaplik kitaplik = new Kitaplik(rafsayisi);
		
		Kitap ktp1, ktp2, ktp3;
		
		ktp1 = new Kitap("Pera Palas'ta Gece Yarisi", "Charles King", "978-605-105-148-2", 1, 1);
        ktp2 = new Kitap("Ali Emiri'nin izinde     ", "M. Serhan Taybi", "978-975-263-956-0", 1, 0);
        ktp3 = new Kitap("Bir Türk Ailesinin Öyküsü", "Orfan Orga", "978-975-289-567-6", 2, 2);
        
        kitaplik.kitapEkle(ktp1);
        kitaplik.kitapEkle(ktp2);
        kitaplik.kitapEkle(ktp3);
        
        kitaplik.gosterRaf(0);
        kitaplik.gosterRaf(1);
        kitaplik.kitapSil(ktp1);
        kitaplik.gosterRaf(1);
        kitaplik.gosterRaf(2);
        
		Scanner giris = new Scanner(System.in);
		int no;
		
		do{
            System.out.println("Raf numarasi giriniz: ");
            no = giris.nextInt();
        }while(!kitaplik.rafKontrol(no));
		
		kitaplik.gosterRaf(no);
		
		giris.close();
		
		
	}

}
