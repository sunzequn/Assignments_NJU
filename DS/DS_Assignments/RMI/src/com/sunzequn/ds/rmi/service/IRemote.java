package com.sunzequn.ds.rmi.service;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.Date;

/**
 * Created by Sloriac on 2016/11/24.
 */
public interface IRemote extends Remote{

    public String generateTime() throws RemoteException;

    public boolean validate(String word) throws RemoteException;

    public void setWord(String word)  throws RemoteException;
}
