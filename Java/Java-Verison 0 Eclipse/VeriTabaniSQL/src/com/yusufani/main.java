package com.yusufani;

import java.sql.Statement;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class main {

	public static void main(String[] args) {
		try(Connection baglanti = DriverManager.getConnection("jdbc:sqlite:test.db");
			Statement statement =baglanti.createStatement();){
			baglanti.setAutoCommit(false); // Bunu yaparsan yaptığın değişikleri elle göndermen gerekir
			statement.execute("CREATE TABLE IF NOT EXISTS kisiler(isim TEXT,email TEXT)");
			statement.execute("INSERT INTO kisiler(isim,email) VALUES ('emre','emre@gmail.com')");
			statement.execute("INSERT INTO kisiler(isim,email) VALUES ('ahmet','ahmet@gmail.com')");
			statement.execute("UPDATE kisiler SET isim='ali' WHERE isim='ahmet'");
			//int etkilenenSatirSayisi =statement.executeUpdate("update kisiler set isim='emre2' where isim='emre'");// Etkilenen satir sayisini yazar.
			//System.out.println(etkilenenSatirSayisi);
			//statement.execute("delete from kisiler where isim='hasan'");// silmek için kullanıyoruz.
			
			ResultSet sonuclar = statement.executeQuery("SELECT * FROM kisiler");//Execute query bir sonuc döndürür
			baglanti.commit(); // Bununla commit ediyoruz.
			while(sonuclar.next()) {
				System.out.println("Kişi adı: "+ sonuclar.getString(1)+ " Email"+sonuclar.getString("email"));
			}
			PreparedStatement preparedStatement = baglanti.prepareStatement("select * from kisiler where isim = ? and email = ? ");//Bir statemnt hazırla ama bağlantılarını verme
			preparedStatement.setString(1, "hasan");
			preparedStatement.setString(2, "hasan@gmail.com");
			sonuclar=preparedStatement.executeQuery();
			while(sonuclar.next()) {
				System.out.println("Kişi adı: "+ sonuclar.getString(1)+ " Email"+sonuclar.getString("email"));
			}
		}catch(SQLException e) {
			e.printStackTrace();
		}

	}

}
