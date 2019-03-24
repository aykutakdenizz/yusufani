package com.yusufani.notdefteri.datamodel;

import java.time.LocalDate;

public class NotOge {
	private String baslik,notDetay;
	private LocalDate bitisTarihi;
	public NotOge(String baslik, String notDetay, LocalDate bitisTarihi) {
		super();
		this.baslik = baslik;
		this.notDetay = notDetay;
		this.bitisTarihi = bitisTarihi;
	}
	public String getBaslik() {
		return baslik;
	}
	public void setBaslik(String baslik) {
		this.baslik = baslik;
	}
	public String getNotDetay() {
		return notDetay;
	}
	public void setNotDetay(String notDetay) {
		this.notDetay = notDetay;
	}
	public LocalDate getBitisTarihi() {
		return bitisTarihi;
	}
	public void setBitisTarihi(LocalDate bitisTarihi) {
		this.bitisTarihi = bitisTarihi;
	}
	public String toString() {
		return baslik;
	}
}
