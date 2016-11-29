package com.sunzequn.ds.rmi.ui;

import javax.swing.*;

/**
 * Created by Sloriac on 16/11/27.
 */
public class ServerUi {

    public static void main(String args[]){

        ServerFrame myFrame = new ServerFrame("服务端");
        myFrame.setSize(400, 100);
        myFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        myFrame.setVisible(true);

    }
}
