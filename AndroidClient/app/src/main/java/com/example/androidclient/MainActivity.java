package com.example.androidclient;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;

public class MainActivity extends AppCompatActivity {
    private Socket sendSocket;
    private BufferedReader networkReader;
    private BufferedWriter networkWriter;
    EditText sendText;
    Button sendBtn;

    private String IP = "155.230.93.237";
    private int PORT = 9898;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        sendText = (EditText) findViewById(R.id.sendText);
        sendBtn = (Button) findViewById(R.id.sendBtn);
        ConnectThread connectThread = new ConnectThread();
        connectThread.start();
    }

    @Override
    protected void onStop() {
        super.onStop();
        try {
            sendSocket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void SendMessage(View view) {
        if (sendText.getText().toString() != null || !sendText.getText().toString().equals("")) {
            new Thread() {
                public void run() {
                    PrintWriter printWriter = new PrintWriter(networkWriter, true);
                    String sendMsg = sendText.getText().toString();
                    printWriter.println(sendMsg);
                }
            }.start();
        }
    }

    class ConnectThread extends Thread {

        public ConnectThread() {
        }

        public void run() {
            try {
                //소켓
                sendSocket = new Socket(IP, PORT);
                //Writer Reader 설정
                networkWriter = new BufferedWriter(new OutputStreamWriter(sendSocket.getOutputStream()));
                networkReader = new BufferedReader(new InputStreamReader(sendSocket.getInputStream()));

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(getApplicationContext(), "Connected", Toast.LENGTH_LONG).show();
                    }
                });

                StartChat.start();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    private Thread StartChat = new Thread() {

        public void run() {
            try {
                String SendMessage;
                while (true) {
                    SendMessage = networkReader.readLine();
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    };
}