package application;
	
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Pos;
import javafx.stage.Stage;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;


public class Main extends Application {
	@Override
	public void start(Stage primaryStage) {
		try {
			//GridPane root = new GridPane();
			
			//scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			Parent root = FXMLLoader .load(getClass().getResource("HBoxOrnek.fxml"));
			Scene scene = new Scene(root,600,400);
			/*primaryStage.setTitle("Merhaba JavaFx");

			Label selam = new Label("Merhhaba Java Fx");
			selam.setTextFill(Color.BLUE);
			selam.setFont(Font.font ("Verdana", 20));
			root.setAlignment(Pos.CENTER);
			root.getChildren().addAll(selam);*/

			primaryStage.setScene(scene);
			primaryStage.show(); 
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		launch(args);
	}
}
