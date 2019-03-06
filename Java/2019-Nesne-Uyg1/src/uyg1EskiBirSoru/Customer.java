package uyg1EskiBirSoru;

import java.util.LinkedList;

public class Customer {
	private String name;
	LinkedList<Product> list;  // Musterinin urunleri linked listede tutuluyor
	private int allCost=0;
	public Customer(String name) {
		this.list = new LinkedList<Product>();
		this.name = name;
	}

	public void UrunEkle(Product product) {
		this.list.add(product);
		this.allCost+=product.getCost();
	}
	public int getAllCost() {
		return allCost;
	}
	public void UrunGoruntule() {
		if ( list == null) System.out.println("Daha urun eklemedeniz");
		else {
			int i=1;
			System.out.println("Num:Isım Marka SKT Fiyat\n");
			for(Product x : list ) {
				System.out.println(i+":"+x.getName()+" "+x.getBrand()+" "+x.getSKT()+" "+x.getCost()+"\n");
				i++;
			}
		}
	}
	public Product Urunara(String productName) {
		for(Product x : list ) {
			if(x.getName().compareTo(productName)== 0) return x;
		}
		return null;
	}
	public void  UrunSil(Product eleman) {
		this.list.remove(eleman);
	}
}
