package com.sunzequn.ds.rmi.service;

import java.net.MalformedURLException;
import java.rmi.AlreadyBoundException;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import org.junit.Test;

/**
 * Created by Sloriac on 2016/11/24.
 */
public class MainApp {

    @Test
    public void serverTest() throws AlreadyBoundException, RemoteException, MalformedURLException {
        TimeServer timeServer = new TimeServer();
        timeServer.server();
        while (true);
    }

    @Test
    public void clientTest() throws RemoteException, NotBoundException, MalformedURLException {
        TimeClient timeClient = new TimeClient();
        timeClient.client();
    }
}
