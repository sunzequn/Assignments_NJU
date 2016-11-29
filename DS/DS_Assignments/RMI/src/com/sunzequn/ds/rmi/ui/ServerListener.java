package com.sunzequn.ds.rmi.ui;

import com.sunzequn.ds.rmi.service.TimeServer;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.rmi.RemoteException;

/**
 * Created by Sloriac on 2016/11/27.
 */
public class ServerListener implements ActionListener {

    private JTextField portTextField;
    private JTextField wordTextField;
    private JButton startButton;
    private JLabel infoLabel;
    private TimeServer server;
    private ServerTimeThread serverTimeThread;

    public ServerListener(JTextField portTextField, JTextField wordTextField, JButton startButton, JLabel infoLabel) {
        this.portTextField = portTextField;
        this.wordTextField = wordTextField;
        this.startButton = startButton;
        this.infoLabel = infoLabel;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        try {
            clear();
            if (startButton.getText().equals("启动")){
                String port = portTextField.getText();
                String word = wordTextField.getText();
                if (!port.equals("") && !word.equals("")){
                    int portNum = Integer.valueOf(port);
                    server = new TimeServer(portNum, word);
                    server.run();
                    startButton.setText("停止");
                    display();
                } else {
                    infoLabel.setText("请输入参数");
                }
            } else {
                server.stop();
                startButton.setText("启动");
                infoLabel.setText("服务已停止");
                serverTimeThread.stop();
            }
        }catch (NumberFormatException ex1){
            ex1.getCause();
            infoLabel.setText("参数错误，请重试！");
        } catch (Exception ex2){
            ex2.getCause();
            infoLabel.setText("端口被占用，请重试！");
        }
    }

    private void clear(){
        infoLabel.setText("");
    }

    private void display() throws RemoteException {
        serverTimeThread = new ServerTimeThread(server, infoLabel);
        serverTimeThread.start();
    }
}
