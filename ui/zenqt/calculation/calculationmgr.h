#ifndef __CALCULATION_MGR_H__
#define __CALCULATION_MGR_H__

#include <QObject>
#include <QThread>
#include <string>
#include <QtWidgets>


class CalcWorker : public QObject
{
    Q_OBJECT
public:
    CalcWorker(QObject* parent = nullptr);

signals:
    void calcFinished(bool, QString);

public slots:
    void run();
};


class CalculationMgr : public QObject
{
    Q_OBJECT
public:
    CalculationMgr(QObject* parent);
    void run();
    void kill();

signals:
    void calcFinished(bool, QString);

private slots:
    void onCalcFinished(bool, QString);

private:
    bool m_bMultiThread;
    CalcWorker* m_worker;
    QThread m_thread;
};

#endif