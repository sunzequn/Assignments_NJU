package com.sunzequn.ds.rmi.ui;

import javax.swing.*;
import java.awt.*;

/**
 * Created by Sloriac on 16/11/27.
 */
public class ClientFrame extends JFrame {


    private Container container;
    private BoxLayout boxLayout;
    private JPanel upperPanel;
    private JPanel bottomPanel;
    private JPanel midPanel;
    private JLabel upperLabel;

    public ClientFrame(String name) throws HeadlessException {
        super(name);
        init();
        initUpperPanel();
        initBottomPanel();
//        initMidPanel();
    }


    private void init() {
        container = getContentPane();
        boxLayout = new BoxLayout(container, BoxLayout.Y_AXIS);
        container.setLayout(boxLayout);
    }


    private void initBottomPanel(){
        bottomPanel = new JPanel();
        bottomPanel.setLayout(new BorderLayout());
        bottomPanel.setPreferredSize(new Dimension(300, 50));
        JButton button = new JButton();
        button.setText("连接");
        bottomPanel.add(button, BorderLayout.SOUTH);
        String inputValue = JOptionPane.showInputDialog("Please input a value");
        this.add(bottomPanel);
    }

    private void initMidPanel(){
        bottomPanel = new JPanel();
        bottomPanel.setLayout(new BoxLayout(bottomPanel, BoxLayout.X_AXIS));
        bottomPanel.setPreferredSize(new Dimension(300, 25));
        JLabel label = new JLabel("链接");
        bottomPanel.add(label);
        add(bottomPanel);
    }

    private void initUpperPanel(){
        upperPanel = new JPanel();
        upperPanel.setLayout(new BorderLayout());
        upperPanel.setPreferredSize(new Dimension(300, 250));
        upperLabel = new JLabel("", SwingConstants.CENTER);
        upperLabel.setText("label");
        upperPanel.add(upperLabel, BorderLayout.CENTER);
        this.add(upperPanel);
    }
}
