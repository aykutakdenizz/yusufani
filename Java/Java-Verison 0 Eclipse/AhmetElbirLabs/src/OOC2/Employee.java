package OOC2;

public class Employee {
	
    private String empId;
    private String Name;
    private int age;
    //Aggregation Java example
    private Address address;

    public String getEmpId() {
            return empId;
    }
    public void setEmpId(String empId) {
            this.empId = empId;
    }
    public String getName() {
            return Name;
    }
    public void setName(String Name) {
            this.Name = Name;
    }
    public int getAge() {
            return age;
    }
    public void setAge(int age) {
            this.age = age;
    }
    public void setAddress(Address address) {
            this.address = address;
    }
    public Address getAddress() {
            return address;
    }
    @Override
    public String toString() {
        return "Employee [empId=" + empId + ", fName=" + Name
        + ", age=" + age + ", address="+ address + "]";
    }
}