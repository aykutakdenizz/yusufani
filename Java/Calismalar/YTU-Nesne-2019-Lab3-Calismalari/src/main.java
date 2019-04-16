import java.util.ArrayList;
import java.util.Iterator;

public class main {
    public static void main(String args[]) {
        String [] isimler  = {"ali","veli","veli","nuriye","veri"};
        String [] soyisimler =  {"aliye","veliye","veliye","nuriyeye","veriye"};
        ArrayList<String> isimListesi = new ArrayList<>();
        ArrayList<String> soyisimListesi = new ArrayList<>();
        for( String s  : isimler) isimListesi.add(s);
        for( String s : soyisimler) soyisimListesi.add(s);
        Iterator<String> iterator = isimListesi.listIterator();
        while(iterator.hasNext()){
            if(soyisimListesi.contains(iterator.next())){
                System.out.println("if ici:"+iterator.next());
                iterator.remove();
            }
        }
    }
}
