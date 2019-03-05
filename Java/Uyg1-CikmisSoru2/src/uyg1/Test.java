package uyg1;

public class Test {
	private int KDV,customerCost;
	public Test (int customerCost) {
		this.customerCost=customerCost;
		this.KDV=customerCost*18/100;
	}
	public int IndirimKontrol() {
		int cost=this.customerCost+this.KDV;
		if (cost>20) cost-=5;
		return cost;
	}
}
