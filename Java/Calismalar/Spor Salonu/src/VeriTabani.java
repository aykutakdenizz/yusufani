import java.sql.*;

public class VeriTabani {
	public static final String DB_NAME = "musteri.db";
	public static final String CONNECTION_STRING = "jdbc:sqlite:"+ DB_NAME;
	public static final String TABLE_MUSTERI="musteri";
	public static final String SUTUN_MUSTERI_ID = "CustomerID";
	public static final String SUTUN_MUSTERI_ADI = "Name";
	public static final String SUTUN_MUSTERI_SOYADI= "Surname";
	public static final String SUTUN_MUSTERI_SIFRESI ="Password";
	public static final String SUTUN_MUSTER_SALON_SAATLERI= "Saatler";
	public static final String SUTUN_MUSTERI_HAREKETLERI1= "Hareketler1";
	public static final String SUTUN_MUSTERI_HAREKETLERI2= "Hareketler2";
	public static final String SUTUN_MUSTERI_HAREKETLERI3= "Hareketler3";
	public static final String SUTUN_MUSTERI_VKI= "VKI";
	public static final String SUTUN_MUSTERI_KAS_ORANI= "KasOrani";
	public Connection baglanti;
	public boolean VeriTabaniniAc() {
		try {
			baglanti=DriverManager.getConnection(CONNECTION_STRING);
			return true;
		}catch(SQLException e) {
			System.out.println("VeriTabanina Baglanilamadi");
			return false;
		}
	}
	public void VeriTabaniniKapa() {
		try {
			if( baglanti != null) {
				baglanti.close();
				System.out.println("Veritabani Basariyla Kapatildi");
			}
		}catch ( SQLException e ) {
			e.printStackTrace();
			System.out.println("Veritabani kapatilamadi");
		}
	}
	public Musteri musteriyiBul(int customerID,String password) {
		StringBuilder sb = new StringBuilder("SELECT * FROM ");
		sb.append(TABLE_MUSTERI);
		try (Statement statement = baglanti.createStatement();
			 ResultSet sonuc = statement.executeQuery(sb.toString())) {
			boolean flag = true;
			while (flag && sonuc.next()) {
				if (customerID == (sonuc.getInt(SUTUN_MUSTERI_ID)) && password.equals(sonuc.getString(SUTUN_MUSTERI_SIFRESI)))
					flag = false;
			}
			if (flag == true) {
				System.out.println("Boyle bir kayit bulunamadi");
				return null;
			} else {
				Musteri x = new Musteri(sonuc.getInt(SUTUN_MUSTERI_ID), sonuc.getString(SUTUN_MUSTERI_ADI), sonuc.getString(SUTUN_MUSTERI_SOYADI), sonuc.getString(SUTUN_MUSTERI_SIFRESI), sonuc.getString(SUTUN_MUSTER_SALON_SAATLERI), sonuc.getString(SUTUN_MUSTERI_HAREKETLERI1), sonuc.getString(SUTUN_MUSTERI_HAREKETLERI2), sonuc.getString(SUTUN_MUSTERI_HAREKETLERI3), sonuc.getDouble(SUTUN_MUSTERI_VKI), sonuc.getDouble(SUTUN_MUSTERI_KAS_ORANI));
				return x;
			}

		} catch (SQLException e) {
			System.out.println("Sorgu Basarisiz");
			e.printStackTrace();
			return null;
		}
	}
	public void musteriDersProgramıGuncelle(Musteri musteri){
		String sorgu = "UPDATE " + TABLE_MUSTERI + " SET " + SUTUN_MUSTERI_HAREKETLERI1+" = ? ,"+SUTUN_MUSTERI_HAREKETLERI2+" = ? ,"+SUTUN_MUSTERI_HAREKETLERI3+ "= ? WHERE " + SUTUN_MUSTERI_ID + " = ? ";
		try(PreparedStatement statement = baglanti.prepareStatement(sorgu)){
			statement.setString(1,musteri.getHareketler1());
			statement.setString(2,musteri.getHareketler2());
			statement.setString(3,musteri.getHareketler3());
			statement.setInt(4,musteri.getId());
			int sonuc = statement.executeUpdate();
			System.out.println("kaç kayıt "+sonuc);
		}catch (SQLException e){
			e.printStackTrace();
		}
	}
}
