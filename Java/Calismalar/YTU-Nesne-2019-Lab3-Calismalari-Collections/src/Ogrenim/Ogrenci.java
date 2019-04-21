public class Ogrenci implements Comparable<Ogrenci>
{
    private int ogreNo;
    private int ogrNot;

    public Ogrenci(int ogreNo, int ogrNot) {
        this.ogreNo = ogreNo;
        this.ogrNot = ogrNot;
    }

    @Override
    public String toString() {
        return "Ogrenci No: " + this.ogreNo + " Ogrenci Not: "+ this.ogrNot;
    }

    @Override
    public int compareTo(Ogrenci o) {
        if (this.ogrNot == o.ogrNot){
            return 0;
        }
        else if ( this.ogrNot > o.ogrNot)
            return 1;
        else return -1;
    }
}
