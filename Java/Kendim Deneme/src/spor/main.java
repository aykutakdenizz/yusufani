package spor;
import java.util.*;
import javax.swing.*;
import javax.swing.JOptionPane;
public class main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String isim=JOptionPane.showInputDialog("Spor Salonu Sistemine Hosgeldiniz!\nLutfen isim giriniz:");
		insan musteri= new insan(isim); 
		JTextField Boy = new JTextField(5);
	    JTextField Yas = new JTextField(5);
	    JTextField kilo = new JTextField(5);
	    JPanel myPanel = new JPanel();
	    myPanel.add(new JLabel("Boy:"));
	    myPanel.add(Boy);
	    myPanel.add(new JLabel("Yas:"));
	    myPanel.add(Yas);
	    myPanel.add(new JLabel("Kilo:"));
	    myPanel.add(kilo);
	    int result = JOptionPane.showConfirmDialog(null, myPanel, 
	               "Lutfen gerekli bosluklari doldurunuz", JOptionPane.OK_CANCEL_OPTION);
	    musteri.setBoy(Integer.parseInt(Boy.getText()));
	    musteri.setYas(Integer.parseInt(Yas.getText()));
	    musteri.setKilo(Integer.parseInt(kilo.getText()));
	    float a=musteri.getBmi();
	    JOptionPane.showMessageDialog(null,a,"Bmi",JOptionPane.WARNING_MESSAGE);
	}

}