package oop;

public class Kitaplik {
	
	Raf raf[];
	
	public Kitaplik(int rafsayi) {
        raf = new Raf[rafsayi];
        for (int i = 0; i < rafsayi; i++) {
            raf[i] = new Raf();
        }
    }
	
	public void kitapEkle(Kitap ktp) {
        raf[ktp.getRafNo()].KitapEkle(ktp);
    }
	
	public Kitap kitapSil(Kitap kitap){
	    return raf[kitap.getRafNo()].kitapSil(raf[kitap.getRafNo()].kitaplar.indexOf(kitap));
	}
	
	public Kitap kitapAra(String kitapAdi){
		for(int i = 0; i < raf.length; i++)
	        for(Kitap kitap:raf[i].kitaplar)
	            if(kitap.getKitapAd().compareTo(kitapAdi) == 0)
	                return kitap;
		return null;
	}
	
	public boolean rafKontrol(int rafno){
		if (raf[rafno] == null) {
			return false;
		}else{
			return true;
		}
	}
	
	public void gosterRaf(int rafno) {
        System.out.println(rafno + ". Raftaki Kitaplar...\nKitap Adı\t\t\tYazar\t\tISBN");
        for(int i = 0; i < raf[rafno].kitaplar.size(); i++){   
            System.out.println(raf[rafno].kitaplar.get(i).getKitapAd() +
            		"\t" + raf[rafno].kitaplar.get(i).getYazar() + "\t" + 
            		raf[rafno].kitaplar.get(i).getISBN());
        }
        System.out.println("\n\n");
    }

}
