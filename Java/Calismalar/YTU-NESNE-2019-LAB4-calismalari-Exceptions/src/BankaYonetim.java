import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class BankaYonetim {
    private int maxMusteriSayisi;
    private int musteriSayisi;
    private ArrayList<Musteri> musteriler;
    public BankaYonetim() {
        musteriler = new ArrayList<>();
        maxMusteriSayisi=2;
        musteriSayisi=0;
    }

    public void musteriEkle(Musteri x) throws MaxMusteriSayisiException {
        if ( musteriSayisi < maxMusteriSayisi){
            musteriler.add(x);
            musteriSayisi++;
            System.out.println("Musteri eklendi");
        }
        else {
            MaxMusteriSayisiException e = new MaxMusteriSayisiException("Daha fazla musteri ekelenemez");
            throw e;
        }
    }
    public void musteriEkle(Musteri [] x) throws MaxMusteriSayisiException {
        Collections.addAll()
        if(x.length+musteriSayisi < maxMusteriSayisi){
            musteriler.addAll(Arrays.asList(x));
            musteriSayisi+=x.length;
        }
        else{
            MaxMusteriSayisiException e = new MaxMusteriSayisiException("Maximun musteri sayisina ulasildi");
            throw  e;
        }
    }

    public ArrayList getMusteriler() {
        return musteriler;
    }

    public void setMusteriler(ArrayList musteriler) {
        this.musteriler = musteriler;
    }

    public int getMusteriSayisi() {
        return musteriSayisi;
    }

    public void setMusteriSayisi(int musteriSayisi) {
        this.musteriSayisi = musteriSayisi;
    }

    public void musteriSil(Musteri x ) throws TaahhutSuresiDolmadiException, YetersizPuanException {
        if( x.getTaaahutluAlısverisSayisi()<= x.getIslemSayisi()) {
            musteriler.remove(x);
            musteriSayisi--;
        }
        else{
            if(x.getFaydalandıgıAyKadarAldigiBonusPuanlar()-2*x.getPuan() > 0){
                musteriler.remove(x);
                musteriSayisi--;
            }
            else {
               throw new YetersizPuanException("mevcut puanlarınız kampanyadan ayrılmak icin yeterli degildir");
            }
            throw  new TaahhutSuresiDolmadiException("Tahattutlu alisveris sayisiniz yeterli degildir.");
        }
    }
    public void listele(){
        for(Musteri x:musteriler) System.out.println("IDNO: "+x.getId()+"Ad soyad : "+x.getAdSoyad()+"Puanlar "+ x.getPuan() + "Taahhut " +x.getTaaahutluAlısverisSayisi() +"\n");
    }
}
