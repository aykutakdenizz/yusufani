package lab2oncesiorneksoru;

public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Daire daire1 = new Daire (1);
		Daire daire2 = new Daire (2);
		IdariPersonel xkisisi = new IdariPersonel(1, "Yusuf Anı", 1000, daire1);
		AkademikPersonel ykisisi= new AkademikPersonel(2, "Hayri Cakir", "Prof", daire2);
		xkisisi.kendiniTanit();
		ykisisi.kendiniTanit();
	}

}
