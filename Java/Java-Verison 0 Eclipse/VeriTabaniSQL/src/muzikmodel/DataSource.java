package muzikmodel;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;

import muzik.Sarkici;

public class DataSource {
	public static final String DB_NAME = "muzik.db";
	public static final String CONNECTION_STRING = "jdbc:sqlite:"+ DB_NAME;
	public static final String TABLO_ALBUM = "album";
	public static final String SUTUN_ALBUM_ID = "albumID";
	public static final String SUTUN_ALBUM_ADI = "albumAdi";
	public static final String SUTUN_ALBUM_SARKICI_ADI= "sarkiciID";
	
	public static final String TABLO_SARKICI = "sarkici";
	public static final String SUTUN_SARKICI_ID = "sarkiciID";
	public static final String SUTUN_SARKICI_ADI = "sarkiciAdi";
	
	public static final String TABLO_SARKI = "sarki";
	public static final String SUTUN_SARKI_ID = "sarkiID";
	public static final String SUTUN_SARKI_ADI = "sarkiAdi";
	public static final String TABLO_SARKI_ALBUMID = "albumID";
	
	public static final int SIRALA_ARTAN=1;
	public static final int SIRALA_AZALAN=2;
	
	
	private Connection baglanti;
	public boolean VeriTabaniniAc() {
		try {
			baglanti = DriverManager.getConnection(CONNECTION_STRING);
			return true;
		}catch (SQLException e ) {
			System.out.println("Veritabanına bağlanılamadı");
			return false;
		}
	}
	public void  VeriTabaniniKapat() {
		try {
			if(baglanti != null) baglanti.close();
				
			
		}catch ( SQLException e) {
			e.printStackTrace();
			System.out.println("Veritabani kapatilamadi");
		}
	}
	public ArrayList<Sarkici> TumSarkicilariGetir(  int siralama) {
		StringBuilder sb = new StringBuilder("SELECT * FROM ");
		sb.append(TABLO_SARKICI);
		if (siralama == SIRALA_ARTAN) {
			sb.append(" ORDER BY ");
			sb.append(SUTUN_SARKICI_ADI);
			sb.append(" ASC");
		}
		else {
			sb.append(" ORDER BY ");
			sb.append(SUTUN_SARKICI_ADI);
			sb.append(" DESC");
		}
		try(Statement statement = baglanti.createStatement();
				ResultSet sonuc = statement.executeQuery(sb.toString())){
			ArrayList<Sarkici> tumSarkicilar = new ArrayList<>();
			while(sonuc.next()) {
				Sarkici sarkici = new Sarkici();
				sarkici.setSarkiciID(sonuc.getInt(SUTUN_SARKICI_ID));
				sarkici.setSarkiciAdi(sonuc.getString(SUTUN_SARKICI_ADI));
				tumSarkicilar.add(sarkici);
			}
			return tumSarkicilar;
		}
		catch (SQLException e) {
			System.out.println("Sorgu başarısız "+ e.getMessage());
			return null;
		}
	}
	}