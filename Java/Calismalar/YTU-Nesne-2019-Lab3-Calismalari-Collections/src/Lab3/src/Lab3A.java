public class Lab3A {

    public static void main(String[] args) {
        Depo d = new Depo();
        Paket pA = new Paket(10);
        Paket pB = new Paket(20);
        Paket pC = new Paket(30);
        Paket pD = new Paket(40);
  
        d.paketYerlestir(pA);
        d.paketYerlestir(pB);
        d.paketYerlestir(pC);
        d.paketYerlestir(pD);
        
        d.kendiniTanit();
        d.paketCikar(pA.getAlanId());
        d.kendiniTanit();
        d.paketCikar(pC.getAlanId());
        d.kendiniTanit();
        d.paketCikar(pD.getAlanId());
        d.kendiniTanit();
        d.paketYerlestir(pD);
        d.kendiniTanit();
        d.paketCikar(pB.getAlanId());
        d.kendiniTanit();
        d.paketYerlestir(pC);
        d.kendiniTanit();        
    } 
}
