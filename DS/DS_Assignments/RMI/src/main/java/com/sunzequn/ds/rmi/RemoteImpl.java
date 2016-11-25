package com.sunzequn.ds.rmi;

import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

/**
 * Created by Sloriac on 2016/11/24.
 *
 */
public class RemoteImpl extends UnicastRemoteObject implements IRemote {

    protected RemoteImpl() throws RemoteException {
    }

    @Override
    public String generateTime() throws RemoteException  {
        return null;
    }
}
