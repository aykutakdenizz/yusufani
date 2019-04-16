package lab2oncesiorneksoru;

public class AkademikPersonel extends Personel {
	private String unvan;

	public AkademikPersonel(int id,String adSoyad,String unvan, Daire ev) {
		super(id,adSoyad,ev);
		this.unvan=unvan;
	}

	public String getUnvan() {
		return unvan;
	}

	public void setUnvan(String unvan) {
		this.unvan = unvan;
	}
	@Override
	public void kendiniTanit() {
		// TODO Auto-generated method stub
		System.out.println("IsD:"+this.getId()+" Ad Soyad : " + this.getAdSoyad() + " Unvan:"+this.getUnvan()+" Kapi No"+this.getDaireKapiNo());
		
	}
	
}
