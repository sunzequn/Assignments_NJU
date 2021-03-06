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
    private JPanel midPanel;
    private JPanel bottomPanel;
    private JTextField portTextField;
    private JTextField wordTextField;
    private JButton connectButton;
    private JLabel infoLabel;
    private JLabel timeLabel;

    public ClientFrame(String name) throws HeadlessException {
        super(name);
        init();
        initUpperPanel();
        initMidPanel();
        initBottomPanel();
    }


    private void init() {
        container = getContentPane();
        boxLayout = new BoxLayout(container, BoxLayout.Y_AXIS);
        container.setLayout(boxLayout);
    }


    private void initUpperPanel(){
        upperPanel = new JPanel();
        upperPanel.setLayout(new BorderLayout());
        upperPanel.setPreferredSize(new Dimension(400, 180));
        timeLabel = new JLabel("...", JLabel.CENTER);
        timeLabel.setFont(new Font("Dialog", 1, 20));
        upperPanel.add(timeLabel, BorderLayout.CENTER);
        this.add(upperPanel);
    }

    private void initMidPanel(){
        midPanel = new JPanel();
        midPanel.setLayout(new BoxLayout(midPanel, BoxLayout.X_AXIS));
        midPanel.setPreferredSize(new Dimension(400, 30));
        JLabel portLabel = new JLabel(" IP:端口号 ");
        midPanel.add(portLabel);

        portTextField = new JTextField();
        midPanel.add(portTextField);

        JLabel wordLabel = new JLabel(" 口令 ");
        midPanel.add(wordLabel);

        wordTextField = new JPasswordField();
        midPanel.add(wordTextField);

        connectButton = new JButton("连接");
        infoLabel = new JLabel("", JLabel.CENTER);
        connectButton.addActionListener(new ClientListener(portTextField, wordTextField, connectButton, infoLabel, timeLabel));
        midPanel.add(connectButton);
        this.add(midPanel);
    }


    private void initBottomPanel(){
        bottomPanel = new JPanel();
        bottomPanel.setLayout(new BorderLayout());
        bottomPanel.setPreferredSize(new Dimension(400, 30));
        bottomPanel.add(infoLabel);

        this.add(bottomPanel);
    }

}
