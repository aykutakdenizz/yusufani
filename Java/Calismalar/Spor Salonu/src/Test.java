public class Test {
    public static void main (String args[]) {
        VeriTabani x = new VeriTabani();
        x.VeriTabaniniAc();
        System.out.println("VeriTabani acildi");
        Musteri a ;
         a= x.musteriyiBul(1,"123456");
        System.out.println(a.getId()+" "+a.getIsim()+" "+a.getHareketler1());
        a.dersProgramiOlustur(x);
        System.out.println(a.getId()+" "+a.getIsim()+" "+a.getHareketler1());
        x.VeriTabaniniKapa();

        System.out.println("VeriTabani kapatildi");
    }
}
