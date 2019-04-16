package lab2oncesiorneksoru;

public class IdariPersonel extends Personel{
	private int hizmetYili;
	public IdariPersonel(int id,String adSoyad,int hizmetYili, Daire ev) {
		super(id,adSoyad,ev);
		this.hizmetYili=hizmetYili;
	}
	@Override
	public void kendiniTanit() {
		// TODO Auto-generated method stub
		System.out.println("IsD:"+this.getId()+" Ad Soyad : " + this.getAdSoyad() + " Hizmet yili:"+this.getHizmetYili()+" Kapi No"+this.getDaireKapiNo());
		
	}
	public int getHizmetYili() {
		return hizmetYili;
	}
	public void setHizmetYili(int hizmetYili) {
		this.hizmetYili = hizmetYili;
	}

}
