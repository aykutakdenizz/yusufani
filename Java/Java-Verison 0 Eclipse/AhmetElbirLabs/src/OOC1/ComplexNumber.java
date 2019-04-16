/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package OOC1;

public class ComplexNumber {

    /**
     * @return the real
     */
    private int real;
    private int imag;
    public int getReal() {
        return real;
    }

    /**
     * @param real the real to set
     */
    public void setReal(int real) {
        this.real = real;
    }

    /**
     * @return the imag
     */
    public int getImag() {
        return imag;
    }

    /**
     * @param imag the imag to set
     */
    public void setImag(int imag) {
        this.imag = imag;
    }

    
    public ComplexNumber(int real, int imag){
        this.real=real;
        this.imag=imag;
    }
    
    public ComplexNumber(ComplexNumber c){
        this(c.real,c.imag);
        //this.real=c.real;
        //this.imag=c.imag;
    }
    
    public ComplexNumber(){
        
    }
    
    public ComplexNumber add(ComplexNumber c1, ComplexNumber c2){
        
        ComplexNumber c=new ComplexNumber();
        c.setReal(c1.getReal() + c2.getReal());
        c.setImag(c1.getImag() + c2.getImag());
        
        return c;
        //return new(c1.real+c2.real , c1.imag+c2.imag);
    }
    public ComplexNumber add(ComplexNumber c1){ //method overloading
        
        ComplexNumber c=new ComplexNumber();
        c.setReal(c1.getReal() + this.getReal());
        c.setImag(c1.getImag() + this.getImag());
        
        return c;
        //return new(c1.real+c2.real , c1.imag+c2.imag);
    }
    public static ComplexNumber multiply(ComplexNumber c1, ComplexNumber c2){
        
        ComplexNumber c=new ComplexNumber();
        c.setReal(c1.real*c2.real-c1.imag*c2.imag);
        c.setImag(c1.real*c2.imag+c1.imag*c2.real);
        
        return c;
        //return new(c1.real+c2.real , c1.imag+c2.imag);
        
        // (a+bi)*(c+di) = ((a*c)-b*d)+(i(a*d+bc))
    }
    
    public String toString(){
        
        if(this.imag<0)
            return this.getReal()+" - "+this.getImag()+"i";
        else 
            return this.getReal()+" + "+this.getImag()+"i";
    }
    
    
    public boolean Equals(ComplexNumber c){
        boolean state=this.imag==c.imag && this.real==c.real;
        return state;
    }
}
