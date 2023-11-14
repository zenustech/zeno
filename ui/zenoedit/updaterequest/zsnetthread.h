#ifndef ZSNETTHREAD_H
#define ZSNETTHREAD_H

#include <QThread>
#include <QObject>

#define URL "https://web.zenustech.com/web-api/version/version/list"
#define GET_VERS "version_list"

class QNetworkReply;

class ZsNetThread : public QThread
{
    Q_OBJECT
public:
    explicit ZsNetThread(QObject *parent = nullptr);
    void setParam(const QString& id, int type, QString url, QByteArray reqData);

protected:
    void run();

private:
    void netGet();

signals:
    void netReqFinish(const QString& data, const QString& id);

private:
    int m_reqType;
    QString m_id;
    QString m_url;
    QByteArray m_reqData;

    bool m_bReport = false;
};

#endif // ZSNETTHREAD_H
