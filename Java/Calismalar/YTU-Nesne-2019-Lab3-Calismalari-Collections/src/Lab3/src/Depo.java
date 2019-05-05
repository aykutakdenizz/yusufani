import java.util.HashMap;

public class Depo {
    HashMap<Integer, Paket> alan;
        
    public Depo(){
        alan = new HashMap<Integer, Paket>();
    }    
   
    public void paketYerlestir(Paket p){
        int i=0;
        while(alan.containsKey(i))
            i++;
		
        alan.put(i,p);
	p.setAlanId(i); 
    }
    
    public void paketCikar(int i){
        alan.remove(i);
        System.out.println("**********************************"+i+" nolu alan boşaltıldı");
    }
   
    public void kendiniTanit(){
        int paketSayisi = alan.size();
        int i=0;
        
        if(paketSayisi == 0)
            System.out.println("Depo BOŞ");            
        
        while(paketSayisi!=0){
            if(!alan.containsKey(i))
                System.out.println(i+" numaralı alan bos");
            else {
                System.out.println(i+" numaralı alanda "+alan.get(i).getAgirlik()+" kiloluk paket");        
                paketSayisi--;
            }
            i++;
        }
        System.out.println("");
    }
}
