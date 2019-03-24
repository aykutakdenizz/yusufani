package com.yusufani.notdefteri.datamodel;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;

import javax.swing.text.html.HTMLDocument.HTMLReader.FormAction;

public class NotData {
	private static NotData instance= new NotData(); // Singeletonda tek bir obje oluşturulup onun üzerinden devam edilir.
	private static String dosyaAdi = "notlarListesi.txt";
	private DateTimeFormatter formatter;
	private ArrayList<NotOge>  notListesi;
	public ArrayList<NotOge> getNotListesi() {
		return notListesi;
	}
	public void setNotListesi(ArrayList<NotOge> notListesi) {
		this.notListesi = notListesi;
	}
	private NotData() { // Bu sınıftan 1 obje üretilmeli 
		formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy");
	}
	public static  NotData getInstance() {
		return instance;
	}
}
