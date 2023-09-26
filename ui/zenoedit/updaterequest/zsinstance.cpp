#include "zsinstance.h"

ZsInstance* ZsInstance::_instance = nullptr;
QMutex  ZsInstance::m_instanceMutex;

ZsInstance *ZsInstance::Instance()
{
    if (_instance == nullptr)
    {
        QMutexLocker locker(&m_instanceMutex);
        if (_instance == nullptr)
        {
            _instance = new ZsInstance;
        }
    }
    return _instance;
}


void ZsInstance::NetRequest(const QString &id, int type, QString url, QByteArray reqData)
{
    ZsNetThread* m_pNetMgr = new ZsNetThread;
    m_pNetMgr->setParam(id, type, url, reqData);
    connect(m_pNetMgr, &ZsNetThread::netReqFinish, this, &ZsInstance::slot_netReqFinish);
    m_pNetMgr->start();
    m_pThreadList.append(m_pNetMgr);
}

void ZsInstance::slot_netReqFinish(const QString &data, const QString &id)
{
    emit sig_netReqFinish(data, id);
    ZsNetThread* mThread = (ZsNetThread*)QObject::sender();
    int n = m_pThreadList.indexOf(mThread);
    mThread->wait();
    if (n != -1)
    {
        m_pThreadList.removeAt(n);
    }
    delete mThread;
    mThread = NULL;
}

ZsInstance::ZsInstance(QObject *parent)
    : QObject{parent}
{

}

ZsInstance::~ZsInstance()
{

}
