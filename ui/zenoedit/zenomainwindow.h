#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <unordered_set>
#include <QtWidgets>
#include "dock/zenodockwidget.h"
#include "panel/zenolights.h"
#include "common.h"


struct ZENO_RECORD_RUN_INITPARAM {
    QString sZsgPath = "";
    bool bRecord = false;
    int iFrame = 0;
    int iSFrame = 0;
    int iSample = 0;
    int iBitrate = 0;
    int iFps = 0;
    QString sPixel = "";
    QString sPath = "";
    QString audioPath = "";
    QString configFilePath = "";
    bool exitWhenRecordFinish = false;
};


class ZenoDockWidget;
class DisplayWidget;
class ZenoGraphsEditor;
class LiveTcpServer;
class LiveHttpServer;
class LiveSignalsBridge;

class ZenoMainWindow : public QMainWindow
{
    Q_OBJECT
public:
    ZenoMainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ZenoMainWindow();
    ZenoGraphsEditor* editor() const { return m_pEditor; }
    bool inDlgEventLoop() const;
    void setInDlgEventLoop(bool bOn);
    TIMELINE_INFO timelineInfo();
    void resetTimeline(TIMELINE_INFO info);
    void doFrameUpdate(int frame);

    ZenoLights* lightPanel = nullptr;
    LiveTcpServer* liveTcpServer;
    LiveHttpServer* liveHttpServer;
    LiveSignalsBridge* liveSignalsBridge;

public slots:
    void openFileDialog();
    void onNewFile();
    bool openFile(QString filePath);
    bool saveFile(QString filePath);
    void saveQuit();
    void save();
    void saveAs();
    void onMaximumTriggered();
    void onSplitDock(bool);
    void onToggleDockWidget(DOCK_TYPE, bool);
    void onDockSwitched(DOCK_TYPE);
    void importGraph();
    void exportGraph();
    void onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    void onPrimitiveSelected(const std::unordered_set<std::string>& primids);
    void updateViewport(const QString& action = "");
    DisplayWidget* getDisplayWidget();
    void onRunFinished();
    void onFeedBack();
    void clearErrorMark();
    void updateLightList();
    void directlyRunRecord(const ZENO_RECORD_RUN_INITPARAM& param);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void init();
    void initMenu();
    void initLive();
    void initDocks();
    void verticalLayout();
    void onlyEditorLayout();
    void writeHoudiniStyleLayout();
    void writeSettings2();
    void readHoudiniStyleLayout();
    void readSettings2();
    void adjustDockSize();
    void recordRecentFile(const QString& filePath);
    QString getOpenFileByDialog();
    QString uniqueDockObjName(DOCK_TYPE type);

    ZenoDockWidget *m_viewDock;
    ZenoDockWidget *m_editor;
    ZenoDockWidget *m_data;
    ZenoDockWidget *m_parameter;
    ZenoDockWidget *m_logger;

    ZenoGraphsEditor* m_pEditor;
    bool m_bInDlgEventloop;
};

#endif
