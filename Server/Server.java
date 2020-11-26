import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.imageio.ImageIO;
 
public class Server implements Runnable { 
    public static final int ServerPORT = 9898;
    public static final String ServerIP = "155.230.93.237";
    public ServerSocket serverSocket;
    public Image recvImg;
    public ObjectOutputStream objOutStream;
    public ObjectInputStream objInputStream;
 
	public static void main(String[] args) {
	   Thread ServerThread = new Thread(new Server());
	   ServerThread.start(); 
	}
   
	@Override
	public void run() {
		//set PATH
		SimpleDateFormat dateFormat = new SimpleDateFormat ("yyyyMMdd_HHmmss");
		Date time = new Date();
		String date = dateFormat.format(time);
		try {
			System.out.println("Connecting...");
			ServerSocket serverSocket = new ServerSocket(ServerPORT); 
			while (true) {
				Socket cilentSocket = serverSocket.accept();
				System.out.println("Connect!");
				
				try {
					//recv img
					objInputStream = new ObjectInputStream(cilentSocket.getInputStream());
					byte[] imageByte = (byte[]) objInputStream.readObject();
					
					ByteArrayInputStream inputStream = new ByteArrayInputStream(imageByte);
					BufferedImage bufferedImage = ImageIO.read(inputStream);
					ImageIO.write(bufferedImage, "jpg", new File("../Images/" + date + ".jpg"));
					System.out.println("Received: 'Image'");
				} catch (Exception e) {
					System.out.println("Error");
					e.printStackTrace();
				}
				
				try {
					String[] command = new String[8];
			        command[0] = "python";
			        command[1] = "/mnt/serverhdd2/jiwook/project/main.py";
			        command[2] = "-m";
			        command[3] = "Test";
			        command[4] = "--checkpoint";
			        command[5] = "/mnt/serverhdd2/jiwook/project/Server/Edit.pt";
			        command[6] = "--data_dir";
			        command[7] = "/mnt/serverhdd2/jiwook/project/Images/" + date + ".jpg";
					String msg = null;
					Process process = Runtime.getRuntime().exec(command);
					BufferedReader stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));
			        BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
			        while ((msg = stdInput.readLine()) != null)
			        	System.out.println(msg);
			        while ((msg = stdError.readLine()) != null)
			          	System.out.println(msg);
			    } catch (Exception e) {
			        e.printStackTrace();
			    }
				
				try {	
					byte[] sendImageByte = Files.readAllBytes(Paths.get("../Images/" + date + "_result.jpg"));					
					//send img
					objOutStream = new ObjectOutputStream(cilentSocket.getOutputStream());
					objOutStream.writeObject(sendImageByte);
				} catch (Exception e) {
					System.out.println("Error");
					e.printStackTrace();
				} finally {
					cilentSocket.close();
					System.out.println("Done");
				}
			}
		} catch (Exception e) {
			System.out.println("Error");
			e.printStackTrace();
		}
		return;
	}
}