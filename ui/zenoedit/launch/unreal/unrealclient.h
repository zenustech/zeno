#ifndef ZENO_UNREALCLIENT_H
#define ZENO_UNREALCLIENT_H

#include <QObject>
#include <QTcpSocket>

class IUnrealLiveLinkClient : public QObject {
    Q_OBJECT

public:
    explicit IUnrealLiveLinkClient(QObject* parent) : QObject(parent) {}
    ~IUnrealLiveLinkClient() override = default;

public:
    virtual void init() = 0;
    virtual void cleanupSocket() = 0;

signals:
    void invalid(IUnrealLiveLinkClient * who);
};

class UnrealLiveLinkTcpClient : public IUnrealLiveLinkClient {
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
    void onError(QAbstractSocket::SocketError error);

private:
    QTcpSocket* m_socket;

};

// TODO: darc support LocalSocket for unix-like OS

#endif //ZENO_UNREALCLIENT_H
