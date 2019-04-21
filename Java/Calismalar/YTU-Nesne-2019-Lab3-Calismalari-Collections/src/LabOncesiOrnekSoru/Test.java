package LabOncesiOrnekSoru;

public class Test {
    public static void main(String[] args) {
        BellekYonetimi d = new BellekYonetimi();
        Process pA = new Process("Process-A");
        Process pB = new Process("Process-B");
        Process pC = new Process("Process-C");
        Process pD = new Process("Process-D");
        d.processYerlestir(pA);
        d.processYerlestir(pB);
        d.processYerlestir(pC);
        d.processYerlestir(pD);
        d.kendiniTanit();
        d.processCikar(2); //2.ADRESTEN ÇIKAR
        d.kendiniTanit();
        d.processCikar(3);
        d.kendiniTanit();
        d.processCikar(1);
        d.kendiniTanit();
        d.processYerlestir(pD);
        d.kendiniTanit();
        d.processYerlestir(pB);
        d.kendiniTanit();

    }
}
