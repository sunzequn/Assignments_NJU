package com.sunzequn.ds.rmi.ui;

import javax.swing.*;

/**
 * Created by Sloriac on 16/11/27.
 */
public class MainUi {

    public static void main(String args[]){

        ClientFrame myFrame = new ClientFrame("Client");
        myFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        myFrame.setSize(300, 300);
        myFrame.setVisible(true);

    }
}
