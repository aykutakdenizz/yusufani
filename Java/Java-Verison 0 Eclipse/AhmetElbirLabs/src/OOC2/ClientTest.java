package OOC2;
public class ClientTest {

public static void main(String[] args) {
    aggregationDemo();
    //compositionDemo();
}

@SuppressWarnings("unused")
private static void compositionDemo() {
    Person person = new Person();
 
    person.setEmpId("1");
    person.setFirstName("Ali");
    
    person.getAddress().setCity("ISTANBUL");
    person.getAddress().setAddressLine1("YILDIZ TEKNIK");
    person.getAddress().setStreet("ESENLER");

    System.out.println(person);
    person = null;
    System.out.println("****After PERSON Deletion*****");
    System.out.println(person);
    if(person !=null)
    System.out.println(person.getAddress());
    else
    System.out.println("Address does not exist..");
}

private static void aggregationDemo() {
    Address address = new Address();
    address.setAddressLine1("YILDIZ TEKNIK");
    address.setCity("ISTANBUL");
    address.setStreet("ESENLER");

    Employee employee = new Employee();
    employee.setEmpId("1");
    employee.setAge(30);
    employee.setName("Ali");
    employee.setAddress(address);

    System.out.println(employee);
    employee = null;
    System.out.println("****After Employee Deletion*****");
    System.out.println("Employee: "+employee);
    System.out.println("Address :"+address);
}
}
