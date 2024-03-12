#ifndef __CALCULATION_MGR_H__
#define __CALCULATION_MGR_H__

#include <QObject>
#include <QThread>
#include <string>
#include <QtWidgets>

class DisplayWidget;

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
    void registerRenderWid(DisplayWidget* pDisp);
    void unRegisterRenderWid(DisplayWidget* pDisp);

signals:
    void calcFinished(bool, QString);

private slots:
    void onCalcFinished(bool, QString);
    void on_render_objects_loaded();

private:
    bool m_bMultiThread;
    CalcWorker* m_worker;
    QThread m_thread;
    QSet<DisplayWidget*> m_registerRenders;
    QSet<DisplayWidget*> m_loadedRender;
};

#endif