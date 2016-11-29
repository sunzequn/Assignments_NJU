package com.sunzequn.ds.rmi.ui;

import com.sunzequn.ds.rmi.service.TimeClient;
import com.sunzequn.ds.rmi.service.TimeServer;

import javax.swing.*;
import java.net.MalformedURLException;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.util.Date;

/**
 * Created by Sloriac on 2016/11/27.
 */
public class ClientTimeThread extends Thread {

    private TimeClient client;
    private JLabel infoLabel;
    private JLabel timeLabel;

    public ClientTimeThread(TimeClient client, JLabel infoLabel, JLabel timeLabel) {
        this.client = client;
        this.infoLabel = infoLabel;
        this.timeLabel = timeLabel;
    }

    @Override
    public void run() {
        try {
            while (true) {
                Long t = new Date().getTime();
                String time = client.getTime();
                if (time != null) {
                    infoLabel.setText("服务已连接，延迟:" + (new Date().getTime() - t) + "毫秒");
                    timeLabel.setText(time);
                } else {
                    infoLabel.setText("连接异常！");
                    timeLabel.setText("...");
                }
                Thread.sleep(1000);
            }

        } catch (Exception e) {
            e.getCause();
        }
    }
}
