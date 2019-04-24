public class Standart extends  Musteri {
    public Standart(int id, String adSoyad, int puan, int taaahutluAlısverisSayisi) {
        super(id, adSoyad, puan*2, taaahutluAlısverisSayisi);
    }

    @Override
    public void alisverisYap(int x) {
        this.setPuan(this.getPuan()+2*x);
        this.setTaaahutluAlısverisSayisi(getTaaahutluAlısverisSayisi()-1);

    }
}
