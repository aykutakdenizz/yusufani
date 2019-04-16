package OOC4;

public class MySql implements IDatabase{
     @Override
    public void AddRecord(Book k) {
        System.out.println(k.getId()+"nolu "+k.getTitle()+ " isimli kitap MySQL ile Kayit Eklendi");
    }

    @Override
    public void DeleteRecord(int id) {
        System.out.println(id+" nolu kitap MySQL ile Kayit Silindi");
    }
}
