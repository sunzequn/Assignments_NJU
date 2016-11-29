package com.sunzequn.ds.rmi.service;

import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.Remote;
import java.rmi.RemoteException;

/**
 * Created by Sloriac on 2016/11/24.
 */
public class TimeClient {

    private String word;
    private String name;

    public TimeClient(int port, String word, String ip) {
        this.word = word;
        this.name = "rmi://" + ip + ":" + port + "/time";
    }

    public boolean run(){
        try {
            IRemote remote = (IRemote) Naming.lookup(name);
            return remote.validate(word);
        } catch (NotBoundException | MalformedURLException | RemoteException e) {
            e.getCause();
            return false;
        }
    }

    public String getTime() {
        try {
            IRemote remote = (IRemote) Naming.lookup(name);
            if (remote.validate(word)){
                return remote.generateTime();
            }
        } catch (NotBoundException | MalformedURLException | RemoteException e) {
            e.getCause();
        }
        return null;
    }
}
