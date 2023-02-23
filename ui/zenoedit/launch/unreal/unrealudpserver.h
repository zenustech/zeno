#ifndef ZENO_UNREALUDPSERVER_H
#define ZENO_UNREALUDPSERVER_H

#include <QObject>
#include <QHostAddress>

class QUdpSocket;

class UnrealUdpServer : public QObject {
    Q_OBJECT

public:
    /**
     * Get static instance of UnrealUdpServer.
     * Thread safe.
     * @return instance
     */
    static UnrealUdpServer& getStaticClass();

    explicit UnrealUdpServer(QObject* parent = nullptr);
    ~UnrealUdpServer() override;

    /**
     * Start the server
     * @param qThread thread that this server running at. Pass nullptr to run on caller event loop.
     * @param inAddress clients from which internet interface are allowed
     * @param inPort binding port, default 23343
     */
    void start(QThread* qThread = nullptr, const QHostAddress& inAddress = QHostAddress::LocalHost, int32_t inPort = 23343);

    /**
     * shutdown and release server instance
     */
    void shutdown();

    /**
     * Check does server valid
     * @return status
     */
    bool isRunning();

private slots:
    void onNewMessage();

private:
    QUdpSocket* m_socket;

};

namespace zeno {
    void startUnrealUdpServer(const QHostAddress& inAddress = QHostAddress::LocalHost, int32_t inPort = 23343);
}

#endif //ZENO_UNREALUDPSERVER_H
