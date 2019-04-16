import java.util.ArrayList;

public class Sepet {
	private String ad;
	private ArrayList<Urun> urun = new ArrayList<Urun>();
	public Sepet(String ad) {
		this.ad = ad;
	}
	public void urunEkle(Urun x) {
		urun.add(x);
	}
	public void urunTanit() {
		System.out.println(this.ad+" Sepetindeki Urunler");
		for (Urun x : urun)
			System.out.println(x.toString());
	}
	public String getAd() {
		return ad;
	}
}
