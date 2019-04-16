package lab2oncesiorneksoru;

import java.util.ArrayList;

public class Daire {
	private int kapiNo;
	private ArrayList<Oda> odalar = new ArrayList<Oda>();
	public Daire (int kapiNo) {
		this.kapiNo=kapiNo;
		Oda oda1 = new Oda("oturma odası");
		Oda oda2 = new Oda("yatak  odası");
		Oda oda3 = new Oda("mutfak");
	}
	public int getKapiNo() {
		return kapiNo;
	}
	public void setKapiNo(int kapiNo) {
		this.kapiNo = kapiNo;
	}
}
