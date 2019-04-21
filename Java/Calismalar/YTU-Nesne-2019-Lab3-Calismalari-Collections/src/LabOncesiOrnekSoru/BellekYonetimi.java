package LabOncesiOrnekSoru;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

public class BellekYonetimi {
    private HashMap<Integer,Process> bellek;
    private Queue<Integer> kuyruk ;
    private int calisanProcessSayisi;

    public BellekYonetimi() {
        bellek= new HashMap<>();
        kuyruk = new LinkedList<>();
        this.calisanProcessSayisi=0;
    }

    public void processYerlestir(Process process){
        if ( kuyruk.size() == 0 ){
            bellek.put(calisanProcessSayisi++,process);
            //System.out.println(bellek.get(calisanProcessSayisi-1).hashCode()); //Hash Code u gormek icin
        }
        else {
            bellek.put(kuyruk.poll(),process);
            calisanProcessSayisi++;
        }
    }
    public void processCikar(int removeIndex){
        bellek.remove(removeIndex);
        kuyruk.add(removeIndex);
        System.out.println("*************************************************"+removeIndex+" nolu adresteki process cikarildi");
    }
    public void kendiniTanit() {
        System.out.println("BOS YERLER : " + kuyruk);
        System.out.println("DOLU YERLER");
        for(Integer x : bellek.keySet()){
            System.out.println(x+"-> "+bellek.get(x).getProcessIsmi()+" numaralı process bellekte");
        }
    }
}
