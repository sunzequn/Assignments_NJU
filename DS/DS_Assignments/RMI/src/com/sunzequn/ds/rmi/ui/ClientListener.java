package com.sunzequn.ds.rmi.ui;

import com.sunzequn.ds.rmi.service.TimeClient;
import com.sunzequn.ds.rmi.service.TimeServer;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.rmi.RemoteException;

/**
 * Created by Sloriac on 2016/11/27.
 */
public class ClientListener implements ActionListener {

    private JTextField portTextField;
    private JTextField wordTextField;
    private JButton connectButton;
    private JLabel infoLabel;
    private JLabel timeLabel;
    private TimeClient client;
    private ClientTimeThread clientTimeThread;

    public ClientListener(JTextField portTextField, JTextField wordTextField, JButton connectButton, JLabel infoLabel, JLabel timeLabel) {
        this.portTextField = portTextField;
        this.wordTextField = wordTextField;
        this.connectButton = connectButton;
        this.infoLabel = infoLabel;
        this.timeLabel = timeLabel;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        try {
            clear();
            if (connectButton.getText().equals("连接")){
                String port = portTextField.getText();
                String word = wordTextField.getText();
                if (!port.equals("") && !word.equals("") && port.contains(":")){
                    String[] params = port.split(":");
                    String ip = params[0];
                    int portNum = Integer.valueOf(params[1]);
                    client = new TimeClient(portNum, word, ip);
                    if (client.run()){
                        connectButton.setText("停止");
                        display();
                    } else {
                        infoLabel.setText("连接失败，请重试！");
                    }
                } else {
                    infoLabel.setText("参数错误，请重试！");
                }
            } else {
                clientTimeThread.stop();
                connectButton.setText("连接");
                infoLabel.setText("连接已终止");
                timeLabel.setText("...");
            }
        } catch (Exception ex2){
            ex2.getCause();
            infoLabel.setText("连接失败，请重试！");
        }
    }

    private void display() throws RemoteException {
        clientTimeThread = new ClientTimeThread(client, infoLabel, timeLabel);
        clientTimeThread.start();
    }

    private void clear(){
        infoLabel.setText("");
    }
}
