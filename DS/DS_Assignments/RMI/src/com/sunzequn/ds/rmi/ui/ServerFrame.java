package com.sunzequn.ds.rmi.ui;

import javax.swing.*;
import java.awt.*;

/**
 * Created by Sloriac on 2016/11/27.
 */
public class ServerFrame extends JFrame {

    private Container container;
    private BoxLayout boxLayout;
    private JPanel upperPanel;
    private JPanel midPanel;
    private JPanel bottomPanel;
    private JTextField portTextField;
    private JTextField wordTextField;
    private JButton startButton;
    private JLabel infoLabel;

    public ServerFrame(String name) throws HeadlessException {
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
        upperPanel.setLayout(new BoxLayout(upperPanel, BoxLayout.X_AXIS));
        upperPanel.setPreferredSize(new Dimension(400, 20));
        this.add(upperPanel);
    }

    private void initMidPanel(){
        midPanel = new JPanel();
        midPanel.setLayout(new BoxLayout(midPanel, BoxLayout.X_AXIS));
        midPanel.setPreferredSize(new Dimension(400, 30));

        JLabel portLabel = new JLabel(" 端口号 ");
        midPanel.add(portLabel);

        portTextField = new JTextField();
        midPanel.add(portTextField);

        JLabel wordLabel = new JLabel(" 口令 ");
        midPanel.add(wordLabel);

        wordTextField = new JPasswordField();
        midPanel.add(wordTextField);

        startButton = new JButton("启动");
        infoLabel = new JLabel("", JLabel.CENTER);
        startButton.addActionListener(new ServerListener(portTextField, wordTextField, startButton, infoLabel));
        midPanel.add(startButton);

        this.add(midPanel);
    }


    private void initBottomPanel(){
        bottomPanel = new JPanel();
        bottomPanel.setLayout(new BorderLayout());
        bottomPanel.setPreferredSize(new Dimension(400, 50));
        bottomPanel.add(infoLabel, BorderLayout.CENTER);
        this.add(bottomPanel);
    }


}
