package Ogrenim;

import java.util.*;

public class main {
    public static void main(String args[]) {
        String [] isimler  = {"ali","veli","veli","nuriye","veri"};
        String [] soyisimler =  {"aliye","veliye","veliye","nuriyeye","veri"};
        ArrayList<String> isimListesi = new ArrayList<>();
        ArrayList<String> soyisimListesi = new ArrayList<>();
        for( String s  : isimler) isimListesi.add(s);
        for( String s : soyisimler) soyisimListesi.add(s);
        Iterator<String> iterator = isimListesi.listIterator();
        /*while(iterator.hasNext()){
            if(soyisimListesi.contains(iterator.next())){
                iterator.remove(); // Soy ismine isim karışmışsa sildi.
            }
        }
        for (String s : isimListesi) System.out.println(s);
        for (String s : soyisimListesi) System.out.println(s);
        isimListesi.retainAll(soyisimListesi); // Ortak olan isimler hariç hepsini Listeden Sil.
        for (String s : isimListesi) System.out.println(s);
        System.out.println("Soyisim Listesi");
        for (String s : soyisimListesi) System.out.println(s);*/

        Vector<Integer> sayilar = new Vector<>(); // Vector kullanımı
        sayilar.add(1);
        sayilar.add(2);
        sayilar.add(3);
        sayilar.add(4);
        System.out.println(sayilar.firstElement());
        System.out.println(sayilar.lastElement());
        for ( int i : sayilar){
            System.out.println(i);
        }
        Enumeration<Integer> e=sayilar.elements();
        while(e.hasMoreElements())
            System.out.println(e.nextElement());

    Stack<Integer> sayilarYigini = new Stack<>(); // Stack kullanımı
        System.out.println("Stack kullanımı \n");
    sayilarYigini.push(1);
        sayilarYigini.push(2);
        sayilarYigini.push(3);
        sayilarYigini.push(4);
        sayilarYigini.push(5);
        System.out.println(sayilarYigini.toString());
    Queue<Integer> kuyruk = new LinkedList<>();
        System.out.println("Kuyruk kullanımı \n");
    kuyruk.add(1);
    kuyruk.offer(5);
        kuyruk.add(3);
        kuyruk.offer(4);
        kuyruk.add(2);
    kuyruk.offer(6);
        System.out.println(kuyruk);
        System.out.println(kuyruk.remove());
        System.out.println(kuyruk);
        System.out.println(kuyruk.poll());
        System.out.println(kuyruk);
    PriorityQueue<Integer> oncelikliKuyruk = new PriorityQueue<>();
        System.out.println("Oncelikli kuyruk kullanımı");
    oncelikliKuyruk.add(2);
        oncelikliKuyruk.add(1);
        oncelikliKuyruk.add(4);
        oncelikliKuyruk.add(3);
        System.out.println(oncelikliKuyruk);
        PriorityQueue<Integer> oncelikliKuyruk2 = new PriorityQueue<>(Collections.reverseOrder());
        System.out.println("Oncelikli kuyruk tersten kullanım");
        oncelikliKuyruk2.add(2);
        oncelikliKuyruk2.add(1);
        oncelikliKuyruk2.add(4);
        oncelikliKuyruk2.add(3);
        System.out.println(oncelikliKuyruk2);

        PriorityQueue<Ogrenci> ogrenciler = new PriorityQueue<>();

        Ogrenci ogr1 = new Ogrenci(16011033 , 100);
        Ogrenci ogr2 = new Ogrenci(16011023 , 67);
        Ogrenci ogr3 = new Ogrenci(16011045 , 88);
        Ogrenci ogr4 = new Ogrenci(16011051 , 71);
        ogrenciler.add(ogr1);
        ogrenciler.add(ogr2);
        ogrenciler.add(ogr3);
        ogrenciler.add(ogr4);
        System.out.println(ogrenciler);
    }
}
