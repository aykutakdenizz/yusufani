package OOC2;

public class Person {

private String empId;
private String firstName;
//Composition example
private Address address;


public Person() {
    address = new Address();
}

public String getEmpId() {
    return empId;
}

public void setEmpId(String empId) {
        this.empId = empId;
}

public String getFirstName() {
        return firstName;
}

public void setFirstName(String firstName) {
        this.firstName = firstName;
}

public Address getAddress() {
        return address;
}

@Override
public String toString() {
        return "Person [empId=" + empId + ", firstName=" + firstName
+ ", address=" + address + "]";
}
}
