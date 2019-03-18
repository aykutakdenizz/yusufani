import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;
public class Main extends Application {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("1");
		launch(args);
	}
	
	@Override
	public void start(Stage primaryStage) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("2");
		primaryStage.setTitle("Merhaba JavaFx"); // Bu benim pencerem
		HBox hBox = new HBox(); // Horizontal Box yani yan yana yerleştiriyor
		Scene scene = new Scene(hBox, 400,300); // pencerenin içine yerleştirdiğimiz sahne 
		Button btn1 = new Button(); // Buton üretiyoruz
		btn1.setText("Buton 1 "); // Butona isim verdik 
		//hBox.getChildren().add(btn1);// Butonu Hbox a yerleştirdik
		Label label1 = new Label("İlk label");
		//hBox.getChildren().add(label1);// Labelı Hbox a yerleştirdik
		hBox.getChildren().addAll(btn1,label1); // İkisini aynı anda yerleştirdik  
		primaryStage.setScene(scene );
		primaryStage.show();
		super.init();
	}
	@Override
	public void init() throws Exception {
		// TODO Auto-generated method stub
		System.out.println("3");
		super.init();
	}
	@Override
	public void stop() throws Exception {
		// TODO Auto-generated method stub
		System.out.println("4");
		super.stop();
	}

}
