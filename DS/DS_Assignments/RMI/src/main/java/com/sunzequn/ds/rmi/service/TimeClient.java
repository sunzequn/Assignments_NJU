package com.sunzequn.ds.rmi.service;

import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;

/**
 * Created by Sloriac on 2016/11/24.
 */
public class TimeClient {

    private IRemote remote;

    public void client() throws MalformedURLException, RemoteException, NotBoundException {
        //在RMI注册表中查找指定对象
        remote = (IRemote) Naming.lookup("rmi://localhost:8888/remote");
        //调用远程对象方法
        System.out.println("client:");
        while (true){
            System.out.println(remote.generateTime());
        }

    }
}
