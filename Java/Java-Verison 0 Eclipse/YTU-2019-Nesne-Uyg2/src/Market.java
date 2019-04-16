import java.util.ArrayList;

public class Market {
	private String ad;
	ArrayList<Sepet> sepet = new ArrayList<Sepet>();
	public Market(String ad) {
		this.ad = ad;
	}
	public void sepetEkle(String ad) {
		sepet.add(new Sepet(ad));
	}
	public void marketTanit() {
		System.out.println("Market adi:"+this.ad + "\nSepet Sayisi: "+sepet.size());
		System.out.println("Sepetler");
		for(Sepet x:sepet)
			x.urunTanit();
	}
	public Sepet getSepet(String ad) {
		for ( Sepet x: sepet)
			if (x.getAd().compareTo(ad) == 0 ) return x;
		System.out.println("Boyle bir sepet bulunamadi");
		return null;
	}
}
