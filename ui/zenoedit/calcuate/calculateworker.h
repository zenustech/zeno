#ifndef __CALCULATE_WORKER_H__
#define __CALCULATE_WORKER_H__

#include <QtWidgets>
#include <zeno/extra/GlobalStatus.h>

class CalculateWorker : public QObject
{
    Q_OBJECT

    enum ProgramState {
        kStopped = 0,
        kRunning,
        kQuiting,
    };

public:
    CalculateWorker(QObject* parent = nullptr);
    void setProgJson(const QString& json);
    ProgramState state() const;

signals:
    void viewUpdated(const QString& hint);
    void errorReported(QString nodeName, QString msg);
    void finished();

public slots:
    void work();
    void stop();

private:
    bool chkfail();
    void reportStatus(zeno::GlobalStatus const& stat);
    bool initZenCache();

    QMutex m_mutex;
    QString progJson;
    ProgramState m_state;
};

#endif