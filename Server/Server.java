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
    public Socket csock;
    public Image recvImg;
 
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
					byte[] imageByte = null;
					ObjectInputStream objInputStream = new ObjectInputStream(cilentSocket.getInputStream());
					imageByte = (byte[]) objInputStream.readObject();
					
					ByteArrayInputStream inputStream = new ByteArrayInputStream(imageByte);
					BufferedImage bufferedImage = ImageIO.read(inputStream);
					ImageIO.write(bufferedImage, "jpg", new File("../Images/" + date));
					System.out.println("S: Received: 'Image'");
					objInputStream.close();

					try {
						String msg = null;
						Process process = Runtime.getRuntime().exec("../python3 main.py -m test --checkpoint checkpoint_legacy/7200.pt --data_dir ./Images/" + date + ".jpg");
						BufferedReader stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));
			            BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
			            while ((msg = stdInput.readLine()) != null)
			                System.out.println(msg);
			            while ((msg = stdError.readLine()) != null)
			            	System.out.println(msg);
			            }catch (IOException e) {
			                e.printStackTrace();
			                System.exit(-1);
			            }
					
					byte[] sendImageByte = Files.readAllBytes(Paths.get("../Images/" + date + "_result.jpg"));
					
					//send img
					ObjectOutputStream objOutStream = new ObjectOutputStream(cilentSocket.getOutputStream());
					objOutStream.writeObject(sendImageByte);
					objOutStream.close();
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
	} 
}