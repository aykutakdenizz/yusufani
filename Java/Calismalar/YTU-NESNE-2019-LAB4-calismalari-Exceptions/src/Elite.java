public class Elite extends Musteri {
    public Elite(int id, String adSoyad, int puan, int taaahutluAlısverisSayisi) {
        super(id, adSoyad, puan*4, taaahutluAlısverisSayisi);
    }
    @Override
    public void alisverisYap(int x) {
        this.setPuan(this.getPuan()+4*x);
        this.setTaaahutluAlısverisSayisi(getTaaahutluAlısverisSayisi()-1);

    }
}
