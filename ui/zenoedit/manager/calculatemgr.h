#ifndef __CALCULATE_MGR_H__
#define __CALCULATE_MGR_H__

#include <QtWidgets>

class CalculateWorker;

class CalculateMgr : public QObject
{
    Q_OBJECT
public:
    CalculateMgr(QObject* parent = nullptr);
    void doCalculate(const QString& json);

private:
    QThread m_thread;
    CalculateWorker* m_worker;
};


#endif