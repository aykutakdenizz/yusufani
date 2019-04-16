package lab2;
import java.util.*;
public class Bolum {
 
private String bolumAdi;
private String adres;
private  int kurulus;
private ArrayList <Calisan> arr= new ArrayList <Calisan>();
private int toplamMaas=0;

public Bolum(String bolumAdi,String adres, int kurulus)
{
    this.bolumAdi=bolumAdi;
    this.adres=adres;
    this.kurulus=kurulus;
}    
  public void calisanEkle(Calisan obj)
  {
      arr.add(obj);   
  }       
  public void calisanToplamMaas()
  {
    for(Calisan eleman: arr){

        toplamMaas += eleman.getMaas();
    }
        
    System.out.println("Bölümün toplam maaşı: "+ Integer.toString( toplamMaas));
  }	
}
