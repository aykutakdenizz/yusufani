package OOC4;

public class MyLibrary {

    public static void main(String[] args) {
        // TODO code application logic here
        Book b1=new Book(1,"ABC");
        Book b2=new Book(2,"XYZ");
        IDatabase db=new MySql();
        db.AddRecord(b1);
        db.AddRecord(b2);
        
        db.DeleteRecord(2);
    }
    
}