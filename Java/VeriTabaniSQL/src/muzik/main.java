package muzik;

import java.util.ArrayList;

import muzikmodel.DataSource;

public class main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DataSource dataSource = new DataSource();
		dataSource.VeriTabaniniAc();
		ArrayList<Sarkici> tumSarkicilar =dataSource.TumSarkicilariGetir(DataSource.SIRALA_ARTAN);
		if (tumSarkicilar == null) System.out.println("Hiç sarkici bulunamadi ");
		else {
			for(Sarkici x : tumSarkicilar) {
				System.out.println("ID : "+ x.getSarkiciID()+" Adi:"+x.getSarkiciAdi());
			}
		}
		dataSource.VeriTabaniniKapat();
	}

}
