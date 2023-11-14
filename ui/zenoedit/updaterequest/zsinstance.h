#ifndef ZSINSTANCE_H
#define ZSINSTANCE_H

#include <QObject>
#include <QMutex>
#include "zsnetthread.h"

class ZsInstance : public QObject
{
    Q_OBJECT
public:
    static ZsInstance* Instance();

    /*! @function
    ********************************************************************************
    parameters: 
              [IN] id                :ID
              [IN] type            :request type 0:netPost
              [IN] url               :url
              [IN] reqHeader  :request header
              [IN] reqData      :request data
    call back ：netReqFinish
    *******************************************************************************/
    Q_INVOKABLE void NetRequest(const QString& id, int type, QString url, QByteArray reqData);

private slots:
    void slot_netReqFinish(const QString& data, const QString& id);

signals:
    void sig_netReqFinish(const QString& data, const QString& id);

private:
    ZsInstance(QObject *parent = nullptr);
    ~ZsInstance();

    static ZsInstance* _instance;
    static QMutex  m_instanceMutex;
    QList<QThread*> m_pThreadList;

};

#endif // ZSINSTANCE_H
