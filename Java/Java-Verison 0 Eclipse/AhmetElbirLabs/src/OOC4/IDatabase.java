/* 1- Sürdürülebilrilik
   2- Sınıflara yetenek kazandırma
   3- Abstract sınıftaki her abstract metodu her alt sınıf 
      kullanmak zorunda değil(çoklu kalıtım sağlar)
*/
package OOC4;

public interface IDatabase {
    
    public void AddRecord(Book k);
    public void DeleteRecord(int id);   
}
