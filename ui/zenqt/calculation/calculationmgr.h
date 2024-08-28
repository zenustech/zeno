#ifndef __CALCULATION_MGR_H__
#define __CALCULATION_MGR_H__

#include <QObject>
#include <QThread>
#include <string>
#include <QtWidgets>
#include "uicommon.h"


class DisplayWidget;

class CalcWorker : public QObject
{
    Q_OBJECT
public:
    CalcWorker(QObject* parent = nullptr);

signals:
    void calcFinished(bool, zeno::ObjPath, QString);
    void nodeStatusChanged(zeno::ObjPath, NodeState);

public slots:
    void run();

private:
    bool m_bReportNodeStatus = true;    //在正常运行模式下，是否发送每个节点的运行状态到前端
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
    bool isMultiThreadRunning() const;

signals:
    void calcFinished(bool, zeno::ObjPath, QString);
    void nodeStatusChanged(zeno::ObjPath, NodeState);

public slots:
    void onPlayTriggered(bool bToggled);
    void onFrameSwitched(int frame);

private slots:
    void onCalcFinished(bool, zeno::ObjPath, QString);
    void onNodeStatusReported(zeno::ObjPath, NodeState);
    void on_render_objects_loaded();
    void onPlayReady();

private:
    bool m_bMultiThread;
    CalcWorker* m_worker;
    QThread m_thread;
    QTimer* m_playTimer;
    QSet<DisplayWidget*> m_registerRenders;
    QSet<DisplayWidget*> m_loadedRender;
};

#endif