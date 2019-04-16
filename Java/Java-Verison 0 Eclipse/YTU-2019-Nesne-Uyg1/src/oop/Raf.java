package oop;

import java.util.LinkedList;

public class Raf {
	
	LinkedList<Kitap> kitaplar;

	public Raf() {
		this.kitaplar = new LinkedList<Kitap>();
	}
	
	public void KitapEkle(Kitap ktp){
        this.kitaplar.add(ktp);
    }
	
	public Kitap kitapSil(int index){
        Kitap kitap;
        kitap = this.kitaplar.remove(index); 
        return kitap;
    }
	
}
