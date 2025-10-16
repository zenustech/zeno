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
    CalcWorker();
    void setCurrentGraphPath(const QString& current_graph_path);

signals:
    void calcFinished(bool, QString, QString, const zeno::render_reload_info&);
    void nodeStatusChanged(QString, QmlNodeRunStatus::Value);

public slots:
    void run();

private:
    bool m_bReportNodeStatus = true;    //在正常运行模式下，是否发送每个节点的运行状态到前端
    QString m_current_graph_path;       //记录当前运行开始时的图层级路径
};


class CalculationMgr : public QObject
{
    Q_OBJECT
public:
    CalculationMgr(QObject* parent);
    Q_PROPERTY(RunStatus::Value runStatus READ getRunStatus WRITE setRunStatus NOTIFY runStatus_changed)
    RunStatus::Value getRunStatus() const;

    Q_PROPERTY(bool autoRun READ getAutoRun WRITE setAutoRun NOTIFY autorun_changed)
    bool getAutoRun() const;
    void setAutoRun(bool autoRun);

    Q_INVOKABLE void run();
    Q_INVOKABLE void kill();
    Q_INVOKABLE void run_and_clean();
    Q_INVOKABLE void clear();
    Q_INVOKABLE void clearNodeObjs(const QModelIndex& nodeIdx);
    Q_INVOKABLE void clearSubnetObjs(const QModelIndex& nodeIdx);

    void registerRenderWid(DisplayWidget* pDisp);
    void unRegisterRenderWid(DisplayWidget* pDisp);
    bool isMultiThreadRunning() const;

signals:
    void calcFinished(bool, QString, QString, const zeno::render_reload_info&);
    void renderRequest(QString);
    void nodeStatusChanged(QString, QmlNodeRunStatus::Value);
    void runStatus_changed();
    void autorun_changed();

public slots:
    void onPlayTriggered(bool bToggled);
    void onFrameSwitched(int frame);

private slots:
    void onCalcFinished(bool, QString, QString, const zeno::render_reload_info& info);
    void onNodeStatusReported(QString, QmlNodeRunStatus::Value);
    void on_render_objects_loaded();
    void onPlayReady();

private:
    void setRunStatus(RunStatus::Value runstatus);

    bool m_bMultiThread;
    QScopedPointer<CalcWorker> m_worker;
    QThread m_thread;
    QTimer* m_playTimer;
    QSet<DisplayWidget*> m_registerRenders;
    QSet<DisplayWidget*> m_loadedRender;
    RunStatus::Value m_runstatus;
};

#endif