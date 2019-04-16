
package OOC3;


public class Retired extends Person {
    
        public Retired(int id, String firstname,String lastname,double credi,int age){
        super(id, firstname,lastname,credi,age);        
    }
        
    public void priceCalculate() {
       super.setCredit(super.getCredit()-0.5);
    }
    
}
