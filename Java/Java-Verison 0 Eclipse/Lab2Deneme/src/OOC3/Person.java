
package OOC3;

public abstract class Person {
    private int idNumber;
    private String name;
    private String surName;
    private double credit;
    private int age;
    
    public Person(int id, String firstname,String lastname,double credi,int age){
        this.idNumber = id;
        this.name = firstname;
        this.surName = lastname;
        this.credit = credi;
        this.age = age;
    }

    public int getIdNumber() {
        return idNumber;
    }

    public void setIdNumber(int idNumber) {
        this.idNumber = idNumber;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSurName() {
        return surName;
    }

    public void setSurName(String surName) {
        this.surName = surName;
    }

    public double getCredit() {
        return credit;
    }

    public void setCredit(double credit) {
        this.credit = credit;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }
    
    public void printAll(){
        System.out.print("Tc Kimlik No:" + idNumber+"\tİsim:" + name+"\tSoyisim:" + surName+"\tKredi:" + credit+"\tYaş:" + age+"\n");
    }
    
    public abstract void priceCalculate();
    
    
    public void addAge(){
            this.age += 1; 
    }
    
}
