
package OOC3;
public class Student extends Person {

    public Student(int id, String firstname,String lastname,double credi,int age){
        super(id, firstname,lastname,credi,age);        
    }

    @Override
    public void priceCalculate() {
       super.setCredit(super.getCredit()-1.25);
    }
    
}
