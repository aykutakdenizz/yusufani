package uyg1KendimDeneme;
public class Kitaplik {
	private Raf raf[];
	public Kitaplik(int rafSayisi) {
		raf = new Raf[rafSayisi]; // Diziyi actik 
		for(int i=0;i<rafSayisi;i++)
			raf[i]=new Raf(); // Yeni raflari olustuduk.
	}
		
	public void kitapEkle(Kitap ktp) {
        raf[ktp.getRafNo()].KitapEkle(ktp);
    }
	public void KitapSil(Kitap kitap) {
		raf[kitap.getRafNo()].KitapSil(raf[kitap.getRafNo()].kitaplar.indexOf(kitap));
	}
	public Kitap kitapAra(String kitapAdi){
		for(int i=0;i<raf.length;i++)
			for(Kitap k:raf[i].kitaplar)
				if(k.getKitapAd().compareTo(kitapAdi) == 0 )
					return k;
		return null;
	}
	public boolean rafKontrol(int rafno){
		if(raf[rafno] == null) return false;
		else return true;
	}
	public void gosterRaf(int rafno) {
		System.out.println(rafno + ". Raftaki Kitaplar...\nKitap Adı\t\t\tYazar\t\tISBN");
		for(Kitap ktp: raf[rafno].kitaplar)
			System.out.println(ktp.getKitapAd()+ "\t"+ktp.getYazar()+"\t"+ktp.getISBN());
	}
}
