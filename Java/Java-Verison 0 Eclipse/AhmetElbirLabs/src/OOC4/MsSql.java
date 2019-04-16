package OOC4;
public class MsSql implements IDatabase {

    @Override
    public void AddRecord(Book k) {
        System.out.println(k.getId()+" nolu "+k.getTitle()+ " isimli kitap MS SQL ile Kayit Eklendi");
    }

    @Override
    public void DeleteRecord(int id) {
        System.out.println(id+"nolu kitap MS SQL ile Kayit Silindi");
    }
}
