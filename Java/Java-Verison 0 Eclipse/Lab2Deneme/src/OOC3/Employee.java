
package OOC3;

public class Employee extends Person {
    
    public Employee(int id, String firstname,String lastname,double credi,int age){
        super(id, firstname,lastname,credi,age);        
    }
    
    public void priceCalculate() {
       super.setCredit(super.getCredit()-1.75);
    }
    
}
