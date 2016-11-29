package com.sunzequn.ds.rmi.service;

import java.net.MalformedURLException;
import java.rmi.AlreadyBoundException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.ExportException;

/**
 * Created by Sloriac on 2016/11/24.
 */
public class TimeServer {

    private IRemote remote;
    private int port;
    private String word;
    private String name;

    public TimeServer(int port, String word) {
        this.port = port;
        this.word = word;
        this.name = "rmi://localhost:" + port + "/time";
    }

    public void run() throws RemoteException, MalformedURLException, AlreadyBoundException, NotBoundException {
        remote = new RemoteImpl();
        remote.setWord(word);
        try {
            LocateRegistry.createRegistry(port);
            Naming.bind(name, remote);
        } catch (ExportException e){
            Naming.rebind(name, remote);
        }
        System.out.println("服务端启动成功！");
    }

    public void stop() throws RemoteException, NotBoundException, MalformedURLException {
        Naming.unbind(name);
        System.out.println("服务端已停止！");
    }

    public IRemote getRemote() {
        return remote;
    }
}
