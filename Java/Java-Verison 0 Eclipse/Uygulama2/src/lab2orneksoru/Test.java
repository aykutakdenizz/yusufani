package lab2orneksoru;
public class Test {
public static void main(String[] args) {
// TODO Auto-generated method stub	
    ArastirmaGorevlisi argor=new ArastirmaGorevlisi("Özgür Cingiz", 29, 6, 2500);
    argor.doktoraDurumGoster("PHD");
    argor.kendiniTanit();

    OgretimUyesi ogrt= new OgretimUyesi("BD", 45, 20, 4000);
    ogrt.setUnvan("Doçent");
    ogrt.kendiniTanit();
    
    Bolum blm= new Bolum("CSE", "iSTANBUL", 1980);
    blm.calisanEkle(ogrt);
    blm.calisanEkle(argor);
    blm.calisanToplamMaas();
}
}
