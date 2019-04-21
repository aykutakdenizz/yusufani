package LabOncesiCalismalar;

import java.util.Collection;
import java.util.HashMap;

public class main {
    public static void main(String[] args) {
        HashMap<Integer, String> tablo = new HashMap<>();
        tablo.put(3,"Yusuf");
        tablo.put(2,"ali");
        tablo.put(1,"Aykut");
        tablo.put(4,"Omer");
        System.out.println(tablo);
        System.out.println(tablo.size());
        Collection<String> degerler = tablo.values();
        System.out.println(degerler);

        /*tablo.clear();
        System.out.println(tablo);*/
        tablo.keySet();
        tablo.entrySet();
        System.out.println(tablo.get(2));
        System.out.println(tablo.containsValue("Ahhmet"));
        System.out.println(tablo.containsValue("Yusuf"));


    }
}
