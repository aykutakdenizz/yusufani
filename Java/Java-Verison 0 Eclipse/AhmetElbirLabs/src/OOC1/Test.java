
package OOC1;

/**
 *
 * @author YTU
 */
public class Test {
    public static void main(String[] args){
    ComplexNumber c1=new ComplexNumber(2,3);
    ComplexNumber c2=new ComplexNumber(c1);
    ComplexNumber c3=new ComplexNumber(2,1);
            
    System.out.println(c1);
    System.out.println(c2);
    
    //c4=c1*c3 -> 1+ 8i
    ComplexNumber c4=ComplexNumber.multiply(c1, c3);
    System.out.println(c4);
    
    ComplexNumber c5=c1.add(c2);
    System.out.println(c5);
    
    ComplexNumber c6= new ComplexNumber();
    c6=c6.add(c1, c3);
    System.out.println(c6);
    
    if(c1==c2)//referans karşılaştırma
        System.out.println("aynı");
    else
        System.out.println("hata");
    
    if(c1.Equals(c2))//içerik karşılaştırma
        System.out.println("aynı");
    else
        System.out.println("hata");
        
    }
    
    
}
