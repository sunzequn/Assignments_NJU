package com.sunzequn.ds.rmi.ui;

import com.sunzequn.ds.rmi.service.TimeServer;

import javax.swing.*;
import java.rmi.RemoteException;

/**
 * Created by Sloriac on 2016/11/27.
 *
 */
public class ServerTimeThread extends Thread {

    private TimeServer server;
    private JLabel jLabel;

    public ServerTimeThread(TimeServer server, JLabel jLabel) {
        this.server = server;
        this.jLabel = jLabel;
    }

    @Override
    public void run() {
        while (true){
            try {
                jLabel.setText("服务启动：" + server.getRemote().generateTime());
                Thread.sleep(1000);
            } catch (RemoteException | InterruptedException e) {
                e.getCause();
            }
        }

    }
}
