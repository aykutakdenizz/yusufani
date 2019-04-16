package lab2kendimdeneme;

import lab2orneksoru.ArastirmaGorevlisi;
import lab2orneksoru.Bolum;
import lab2orneksoru.OgretimUyesi;

public class Test {
public static void main(String[] args) {
// TODO Auto-generated method stub	
    ArastirmaGorevlisi argor=new ArastirmaGorevlisi("mmm Cingiz", 29, 6, 2500);
    argor.doktoraDurumGoster("PHD");
    argor.kendiniTanit();

    OgretimUyesi ogrt= new OgretimUyesi("BD", 45, 20, 4000);
    ogrt.setUnvan("Docent");
    ogrt.kendiniTanit();
    
    Bolum blm= new Bolum("CSE", "iSTANBUL", 1980);
    blm.calisanEkle(ogrt);
    blm.calisanEkle(argor);
    blm.calisanToplamMaas();
}
}
