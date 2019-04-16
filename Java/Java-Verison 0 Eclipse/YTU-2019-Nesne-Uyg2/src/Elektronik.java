public class Elektronik extends Urun {

	public  Elektronik(int id, String urunAdi, double fiyat) {
		super(id, urunAdi, fiyat);
		// TODO Auto-generated constructor stub
		
	}

	@Override
	public void fiyatGuncelle() {
		// TODO Auto-generated method stub
		this.setFiyat(this.getFiyat()*1.18);
	}
	
}
