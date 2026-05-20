#ifndef ZENO_APIMGR_H
#define ZENO_APIMGR_H

#include <QObject>
#include <QString>
#include <atomic>
#include <thread>

class ZenoApplication;

class ApiMgr : public QObject
{
    Q_OBJECT

public:
    explicit ApiMgr(ZenoApplication* app, QObject* parent = nullptr);
    ~ApiMgr();

    void start();
    void stop();

    void sendResponse(const QString& response);

    bool isRunning() const { return m_bRunning; }

signals:
    void commandReceived(const QString& command);

public slots:
    void handleCommand(const QString& command);

private:
    void inputListenerThread();

    ZenoApplication* m_app;
    std::atomic<bool> m_bRunning;
    std::atomic<bool> m_running{true};
    std::thread* m_pThread;
};

#endif // ZENO_APIMGR_H