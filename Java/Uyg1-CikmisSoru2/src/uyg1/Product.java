package uyg1;

public class Product {
	private String name,brand,SKT;
	private int cost;
	public Product(String name, String brand ,String SKT , int cost ){
		this.name=name;
		this.brand=brand;
		this.SKT=SKT;
		this.cost=cost;
	}
	public String getName() {
		return name;
	}
	public String getBrand() {
		return brand;
	}
	public String getSKT() {
		return SKT;
	}
	public int getCost() {
		return cost;
	}
}
