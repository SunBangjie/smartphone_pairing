package com.example.toyapplicationv2;

import android.os.AsyncTask;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;

public class MessageSender extends AsyncTask<String, Void, Void> {

    private DataOutputStream dos;
    private String IP;
    private int PORT;

    public MessageSender(String ip, int port) {
        super();
        IP = ip;
        PORT = port;
    }

    @Override
    protected Void doInBackground(String... voids) {
        String message = voids[0];
        try {
            Socket s = new Socket(IP, PORT);
            PrintWriter pw = new PrintWriter(s.getOutputStream());
            pw.write(message);
            pw.flush();
            pw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
