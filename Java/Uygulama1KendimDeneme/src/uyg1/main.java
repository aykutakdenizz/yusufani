package uyg1;

import java.util.Scanner;
public class main {

	public static void main( String[] args) {
		int rafSayisi =5;
		Kitaplik kitaplik = new Kitaplik(rafSayisi);
		int secim;
		Scanner in = new Scanner(System.in);
		boolean exit=true;
		while (exit) {
			System.out.println("1->Kitap Ekle\n2->Kitap Sil\n3->Raftaki kitaplari göster\n4->cikis");
			secim = in.nextInt();
			switch (secim) {
			case 1:{
				String ad,yazar,ibsn;
				int rafNo,sira;
				in.nextLine();
				System.out.println("Lutfen kitap ismi giriniz:");
				ad=in.nextLine();
				System.out.println("Lutfen kitabin yazarini giriniz:");
				yazar=in.nextLine();
				System.out.println("Lutfen kitabin ISBN numarasini giriniz:");
				ibsn=in.nextLine();
				System.out.println("Lutfen kitabin koymak istediginiz raf numarasini giriniz:");
				rafNo=in.nextInt();
				System.out.println("Lutfen kitabin koymak istediginiz raftaki sira numarasini giriniz:");
				sira=in.nextInt();
				Kitap ktp = new Kitap(ad,yazar,ibsn,rafNo,sira);
				if (ktp != null) {
					kitaplik.kitapEkle(ktp);
					System.out.println("Kitap basariyla olusturuldu.");
				}
				else System.out.println("Kitap olusturulamadi.");
				
				break;
			}
			case 2:{
				System.out.println("Lutfen silmek istediginiz kitabin adinizi giriniz:");
				String silinecekKitap;
				in.nextLine();
				silinecekKitap=in.nextLine();
				kitaplik.KitapSil(kitaplik.kitapAra(silinecekKitap));
				break;
			}
			case 3:{
				System.out.println("Lutfen bir raf numarasi tuslayiniz:");
				kitaplik.gosterRaf(in.nextInt());
				break;
			}
			case 4:{
				exit=false;
			}
			
			}
		
		}
        in.close();
	}

}
