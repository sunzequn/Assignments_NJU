package com.sunzequn.ds.rmi.service;

import java.net.MalformedURLException;
import java.rmi.AlreadyBoundException;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;

/**
 * Created by Sloriac on 2016/11/24.
 */
public class TimeServer {

    private IRemote remote;

    public void server() throws RemoteException, MalformedURLException, AlreadyBoundException {
        remote = new RemoteImpl();
        //远程对象注册表实例
        LocateRegistry.createRegistry(8888);
        //把远程对象注册到RMI注册服务器上
        Naming.bind("rmi://localhost:8888/remote", remote);
        System.out.println("server:对象绑定成功！");
    }
}
