package com.example.androidclient;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

public class MainActivity extends AppCompatActivity {
    private Socket socket;
    private ObjectInputStream objInputStream;
    private ObjectOutputStream objOutStream;
    ImageView sendImgView, recvImgView;
    Button imgBtn, sendBtn;
    private byte[] byteArray;
    Bitmap recvImgBitmap;

    private String IP = "155.230.93.237";
    private int PORT = 9898;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imgBtn = (Button) findViewById(R.id.imgBtn);
        sendBtn = (Button) findViewById(R.id.sendBtn);
        sendImgView = (ImageView) findViewById(R.id.sendImg);
        recvImgView = (ImageView) findViewById(R.id.recvImg);
        ConnectThread connectThread = new ConnectThread();
        connectThread.start();
    }

    @Override
    protected void onStop() {
        super.onStop();

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            socket.close();
            objOutStream.close();
            objInputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //이미지 불러오기
    public void setImg(View view) {
        Intent imgIntent = new Intent(Intent.ACTION_GET_CONTENT);
        imgIntent.setType("image/*");
        if (imgIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(imgIntent, 1);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {
            try {
                InputStream inStream = getContentResolver().openInputStream(data.getData());
                Bitmap sendImgBitmap = BitmapFactory.decodeStream(inStream);
                inStream.close();
                sendImgView.setImageBitmap(sendImgBitmap);

                ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                sendImgBitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                byteArray = byteArrayOutputStream.toByteArray();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    //이미지 보내기
    public void SendIMG(View view) {
        Toast.makeText(getApplicationContext(), "잠시만 기다려주세요..", Toast.LENGTH_LONG).show();
        new Thread() {
            public void run() {
                //이미지 전송
                try {
                    objOutStream.writeObject(byteArray);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }.start();
    }

    //이미지 받기
    private Thread recvImgThread = new Thread() {
        public void run() {
            try {
                byte[] inputData = (byte[]) objInputStream.readObject();
                recvImgBitmap = BitmapFactory.decodeByteArray(inputData, 0, inputData.length);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        recvImgView.setImageBitmap(recvImgBitmap);
                    }
                });
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    };

    //소켓 연결
    class ConnectThread extends Thread {

        public ConnectThread() {
        }

        public void run() {
            try {
                //소켓 설정
                socket = new Socket(IP, PORT);
                objOutStream = new ObjectOutputStream(socket.getOutputStream());
                objInputStream = new ObjectInputStream(socket.getInputStream());
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(getApplicationContext(), "Connected", Toast.LENGTH_LONG).show();
                    }
                });
                recvImgThread.start();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
}