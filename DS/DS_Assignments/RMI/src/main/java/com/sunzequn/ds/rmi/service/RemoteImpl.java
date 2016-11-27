package com.sunzequn.ds.rmi.service;

import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.Calendar;

/**
 * Created by Sloriac on 2016/11/24.
 *
 */
public class RemoteImpl extends UnicastRemoteObject implements IRemote {

    protected RemoteImpl() throws RemoteException {
    }

    @Override
    public String generateTime() throws RemoteException  {
        Calendar calendar = Calendar.getInstance();
        int year = calendar.get(Calendar.YEAR);
        int month = calendar.get(Calendar.MONTH);
        int date = calendar.get(Calendar.DATE);
        int hour = calendar.get(Calendar.HOUR_OF_DAY);
        int minute = calendar.get(Calendar.MINUTE);
        int second = calendar.get(Calendar.SECOND);
        int millisecond = calendar.get(Calendar.MILLISECOND);
        return (year + "/" + month + "/" + date + " " +hour + ":" +minute + ":" + second + ":" + millisecond);
    }
}
