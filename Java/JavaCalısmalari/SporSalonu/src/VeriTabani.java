import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class VeriTabani {
	public static final String DB_NAME = "musteriler.db";
	public static final String CONNECTION_STRING = "jdbc:sqlite:"+ DB_NAME;
	public static final String TABLE_MUSTERI="musteri";
	public static final String SUTUN_MUSTERI_ID = "CustomerID";
	public static final String SUTUN_MUSTERI_ADI = "Name";
	public static final String SUTUN_MUSTERI_SOYADI= "Surname";
	public static final String SUTUN_SALON_SAATLERI= "Saatler";
	public static final String SUTUN_MUSTERI_HAREKETLERI= "Hareketler";
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
	public void e
}
