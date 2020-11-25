package com.example.androidclient;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
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
    Image sendImg, recvImg;
    ImageView sendImgView, recvImgView;
    Button imgBtn, sendBtn;
    private byte[] byteArray;

    private String IP = "155.230.93.237";
    private int PORT = 9898;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imgBtn = (Button) findViewById(R.id.imgBtn);
        sendBtn = (Button) findViewById(R.id.sendBtn);
        ConnectThread connectThread = new ConnectThread();
        connectThread.start();
    }

    @Override
    protected void onStop() {
        super.onStop();
        try {
            socket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //이미지 불러오기
    public void setImg(View view) {
        Intent imgIntent = new Intent();
        imgIntent.setType("image/*");
        imgIntent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(imgIntent, 1);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1) {
            if (resultCode == RESULT_OK) {
                try {
                    InputStream inStream = getContentResolver().openInputStream(data.getData());
                    Bitmap sendImgBitmap = BitmapFactory.decodeStream(inStream);
                    inStream.close();
                    sendImgView.setImageBitmap(sendImgBitmap);

                    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    sendImgBitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);
                    byteArray = byteArrayOutputStream.toByteArray();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    //이미지 보내기
    public void SendMessage(View view) {
        new Thread() {
            public void run() {
                //이미지 전송
                try {
                    objOutStream = new ObjectOutputStream(socket.getOutputStream());
                    objOutStream.writeObject(byteArray);
                    objOutStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                recvImgThread.start();
            }
        }.start();
    }

    //이미지 받기
    private Thread recvImgThread = new Thread() {

        public void run() {
            try {
                objInputStream = new ObjectInputStream(socket.getInputStream());
                byte[] inputData = (byte[]) objInputStream.readObject();
                objInputStream.close();
                Bitmap recvImgBitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
                recvImgView.setImageBitmap(Bitmap.createScaledBitmap(recvImgBitmap, recvImgView.getWidth(), recvImgView.getHeight(), false));
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
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(getApplicationContext(), "Connected", Toast.LENGTH_LONG).show();
                    }
                });
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }


}