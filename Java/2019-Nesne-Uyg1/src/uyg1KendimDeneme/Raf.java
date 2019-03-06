package uyg1;

import java.util.LinkedList;

public class Raf {
	LinkedList<Kitap> kitaplar;// Her rafta bir linkli liste yapisi varmis gibi dusunduk
	public Raf (){
		this.kitaplar = new LinkedList<Kitap>(); // Her raf olusturuldugunda bir linkli liste olusması lazim
		
	}
	public void KitapEkle (Kitap kitap) {
		this.kitaplar.add(kitap); // kitabın ismini add komutu ile ekledik
	}
	public Kitap KitapSil( int index) {
		Kitap kitap = this.kitaplar.remove(index); // Belirtilen classdaki linked listte, verilen index numarasındaki kitapı sildik
		return kitap; // Geri dondurmemiz bizden istenmis o yüzden kitapı geri döndürdük.
	}
}
