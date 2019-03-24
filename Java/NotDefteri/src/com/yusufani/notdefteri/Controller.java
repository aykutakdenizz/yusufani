package com.yusufani.notdefteri;

import java.awt.Label;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;

import com.yusufani.notdefteri.datamodel.NotOge;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.fxml.FXML;
import javafx.scene.control.ListView;
import javafx.scene.control.TextArea;

public class Controller {
	@FXML
	private ListView lvNotlarListesi; // Notların listesini arayüzde tuttuğumuz yapı
	@FXML
	private javafx.scene.control.Label lBitisTarihi; //Bitiş tarihih için oluşturduğgumuz label
	private ArrayList<NotOge> notlarListesi; // Notları tuttuğumuz arraylist
	@FXML
	private TextArea taNotDetay;   // Notların detaylarınu yazdırdığımız arayüz yapısı
	@FXML
	public void initialize() { // Otomatik olarak tanımlı bir method 
		NotOge not1 = new NotOge("Hediye Al", "Anneler günü için alışverişe çık",LocalDate.of(2019, 5, 17));
		NotOge not2 = new NotOge("Eczaneye git", "D vitamini hapını al",LocalDate.of(2019, 5, 23));
		NotOge not4 = new NotOge("Detarjan al", "Evde Çamaşır detaranı bitti",LocalDate.of(2019, 5, 4));
		NotOge not3= new NotOge("Machine Learning kurusnu tamamla","Machine önemli",LocalDate.of(2019, 6,21 ));
		NotOge not5 = new NotOge("Diyet yap", "Sağlıklı yaşam adamdır",LocalDate.of(2019, 9, 11));
		
		notlarListesi= new ArrayList<>();
		notlarListesi.add(not1);
		notlarListesi.add(not2);
		notlarListesi.add(not3);
		notlarListesi.add(not4);
		notlarListesi.add(not5);
		lvNotlarListesi.getSelectionModel().selectedItemProperty().addListener(new ChangeListener() {

			@Override
			public void changed(ObservableValue observable, Object oldValue, Object newValue) {
				// TODO Auto-generated method stub
				if (newValue != null ) {
					NotOge secilenNot= (NotOge) lvNotlarListesi.getSelectionModel().getSelectedItem();
					taNotDetay.setText(secilenNot.getNotDetay());
					DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy");// Tarihi Türkiyeye göre ayarladık
					lBitisTarihi.setText(secilenNot.getBitisTarihi().format(formatter) );
				}
				System.out.println("tETİKLENDİ");
			}
		});
		lvNotlarListesi.getItems().setAll(notlarListesi);
		lvNotlarListesi.getSelectionModel().selectFirst();
	}
	/*
	@FXML
	public void listViewSecim() {
		NotOge secilenNot= (NotOge) lvNotlarListesi.getSelectionModel().getSelectedItem();
		/*StringBuilder sb = new StringBuilder(secilenNot.getNotDetay());
		sb.append("\n\n\n");
		sb.append(secilenNot.getBitisTarihi().toString());
		taNotDetay.setText(sb.toString());*/
		/*taNotDetay.setText(secilenNot.getNotDetay());
		lBitisTarihi.setText(secilenNot.getBitisTarihi().toString());
	}*/
}
