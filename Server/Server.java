import java.awt.Image;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
 
public class Server implements Runnable { 
    public static final int ServerPORT = 9898;
    public static final String ServerIP = "155.230.93.237";
    public ServerSocket serverSocket;
    public Socket csock;
    public Image recvImg, sendImg;
 
   public static void main(String[] args) { 
        Thread ServerThread = new Thread(new Server());
        ServerThread.start(); 
    }
   
   @Override
   public void run() {	
       try {
           System.out.println("Connecting...");
           ServerSocket serverSocket = new ServerSocket(ServerPORT); 
           while (true) {
               Socket cilentSocket = serverSocket.accept();
               System.out.println("Connect!");
               try {
                   //�̹��� ��ü ����
               	ObjectInputStream objInputStream = new ObjectInputStream(cilentSocket.getInputStream());
               	recvImg = (Image)objInputStream.readObject();
                   System.out.println("S: Received: 'Image'");
                   objInputStream.close();
                   
                   //���̽� ���� �ڵ� �߰� �ʿ�
                   //���̽� �ڵ� ���� ����� ����ϰ� �ִٰ� recvImg�� ImageIO.read(new File(<image file>));
                   //�н� �����ؼ� ��ü ���� �ؾ���.
                   
                   //�̹��� ��ü ����
                   ObjectOutputStream objOutStream = new ObjectOutputStream(cilentSocket.getOutputStream());
                   objOutStream.writeObject(sendImg);
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