#ifndef ZENO_UNREALCLIENT_H
#define ZENO_UNREALCLIENT_H

#include <QObject>
#include <QTcpSocket>

class UnrealLiveLinkClient : public QObject {
    Q_OBJECT

public:
    explicit UnrealLiveLinkClient(QObject* parent) : QObject(parent) {}
    ~UnrealLiveLinkClient() override = default;

public:
    virtual void init() = 0;
    virtual void cleanupSocket() = 0;

signals:
    void invalid(UnrealLiveLinkClient* who);
};

class UnrealLiveLinkTcpClient : public UnrealLiveLinkClient {
    Q_OBJECT

public:
    void init() override;
    void cleanupSocket() override;

    explicit UnrealLiveLinkTcpClient(QObject* parent, QTcpSocket* inTcpSocket);
    ~UnrealLiveLinkTcpClient() override;

  private:

private slots:
    void onSocketClosed();
    void onSocketReceiveData();

private:
    QTcpSocket* m_socket;

};

#endif //ZENO_UNREALCLIENT_H
