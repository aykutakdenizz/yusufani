public class Gida extends Urun {

	public Gida(int id, String urunAdi, double fiyat) {
		super(id, urunAdi, fiyat);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void fiyatGuncelle() {
		// TODO Auto-generated method stub
		this.setFiyat(this.getFiyat()*1.08);
	}

	
}
