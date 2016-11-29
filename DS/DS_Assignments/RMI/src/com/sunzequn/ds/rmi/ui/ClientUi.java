package com.sunzequn.ds.rmi.ui;

import javax.swing.*;

/**
 * Created by Sloriac on 16/11/27.
 */
public class ClientUi {

    public static void main(String args[]){

        ClientFrame myFrame = new ClientFrame("客户端");
        myFrame.setSize(400, 260);
        myFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        myFrame.setVisible(true);

    }
}
