package uyg1EskiBirSoru;

import java.util.Scanner;

public class main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Product urunler[] = new Product [6];
		urunler[0] = new Product("A urunu","A marka","12/12/2222",5);
		urunler[1] = new Product("B urunu","B marka","12/12/2222",2);
		urunler[2] = new Product("C urunu","C marka","12/12/2222",2);
		urunler[3] = new Product("D urunu","D marka","12/12/2222",5);
		urunler[4] = new Product("E urunu","E marka","12/12/2222",5);
		urunler[5] = new Product("F urunu","F marka","12/12/2222",51);
		boolean exit = true;
		Scanner in = new Scanner(System.in);
		int secim;
		System.out.println("Lutfen isminizi giriniz:");
		String name = in.nextLine();
		Customer customer = new Customer(name);
		while (exit) {
			System.out.println("1->Urun al\n2->Sepeti Goruntule\n3->Sepetten Urun cikar\n4->Alisverisi tamamla");
			secim=in.nextInt();
			switch(secim) {
			case 1:{
				int i=1;
				System.out.println("Num:Isım Marka SKT Fiyat\n");
				for(Product x : urunler) {
					System.out.println(i+":"+x.getName()+" "+x.getBrand()+" "+x.getSKT()+" "+x.getCost()+"\n");
					i++;
				}
				System.out.println("Lutfen almak istediginiz urunun numarasini tuslayiniz:");
				secim = in.nextInt();
				customer.UrunEkle(urunler[secim]);
				break;
			}
			case 2:{
				customer.UrunGoruntule();
				break;
			}
			case 3:{
				System.out.println("Lutfen silmek istediginiz urunun adinizi giriniz:");
				String silinecekUrun;
				in.nextLine();
				silinecekUrun=in.nextLine();
				Product tmp =customer.Urunara(silinecekUrun);
				if ( tmp != null) customer.UrunSil(tmp);
				else System.out.println("Boyle bir urun bulunamadi");
				break;
			}
			case 4: {
				Test test = new Test(customer.getAllCost());
				int alisverissonu=test.IndirimKontrol();
				System.out.println("Alisveris Tutari "+ alisverissonu);
				System.out.println("Yeni musteri girilsin mi :? true->evet false-> hayir");
				exit=in.nextBoolean();
				if (exit == true) {
					
					System.out.println("Lutfen isminizi giriniz:");
					in.nextLine();
					 name = in.nextLine();
					 customer = new Customer(name);
				}
				else {
					exit=false;
				}
			}
			}
		}
		in.close();
	}

}
