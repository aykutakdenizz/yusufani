
package OOC3;
public class AkbilSistemi {
    public static void main(String[] args) {
        // TODO code application logic here
        int n=4;
        Person akbil[] = new Person[n];
        
        akbil[0]=new Student(123456,"Ahmet","Keskin",10,25);
        akbil[1]=new Student(654321,"Ahmet","Ozturk",10,29);
        akbil[2] = new Employee(789123, "Hatice", "Bayrak", 100, 59);
        akbil[3] = new Retired(678345, "Ayse", "Demir", 100, 61);
         
        System.out.println("İlk çıktılar");
        for(int i=0;i<n;i++){
            akbil[i].printAll();
        }
        System.out.println("Otobüse Biniş-1");
         
        for(int i=0;i<n;i++){
            akbil[i].priceCalculate();
        }
         
        for(int i=0;i<n;i++){
            akbil[i].printAll();
        }
         
        System.out.println("Yaş Artışı");
        for(int i=0;i<n;i++){
             akbil[i].addAge();
             //BURASI SINIF GEÇİŞİNDE KULLANILABİR
            if(akbil[i].getAge()>=30 && akbil[i].getAge()<60)
                akbil[i]=new Employee(akbil[i].getIdNumber(),akbil[i].getName(),akbil[i].getSurName(),akbil[i].getCredit(),akbil[i].getAge());
            else if(akbil[i].getAge()>=60)
                akbil[i]=new Retired(akbil[i].getIdNumber(),akbil[i].getName(),akbil[i].getSurName(),akbil[i].getCredit(),akbil[i].getAge());
        }
        
        System.out.println("Otobüse Biniş-2");   
        for(int i=0;i<n;i++){
             akbil[i].priceCalculate();
        }
         
        for(int i=0;i<n;i++){
             akbil[i].printAll();
        }          
    }
}
