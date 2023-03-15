#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <unordered_set>
#include <QtWidgets>
#include "dock/zenodockwidget.h"
#include "dock/ztabdockwidget.h"
#include "panel/zenolights.h"
#include "common.h"
#include "layout/winlayoutrw.h"
#include "cache/zcachemgr.h"


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
class ZTimeline;
class LiveTcpServer;
class LiveHttpServer;
class LiveSignalsBridge;
class ViewportWidget;

namespace Ui
{
    class MainWindow;
}

class ZenoMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    ZenoMainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ZenoMainWindow();
    bool inDlgEventLoop() const;
    void setInDlgEventLoop(bool bOn);
    TIMELINE_INFO timelineInfo();
    void setAlways(bool bAlways);
    bool isAlways() const;
    void resetTimeline(TIMELINE_INFO info);
    ZTimeline* timeline() const;
    QVector<DisplayWidget*> viewports() const;
    ZenoGraphsEditor* getAnyEditor() const;
    void dispatchCommand(QAction* pAction, bool bTriggered);
    std::shared_ptr<ZCacheMgr> cacheMgr() const;

    void doFrameUpdate(int frame);
    void sortRecentFile(QStringList &lst);

    QLineEdit* selected = nullptr;
    ZenoLights* lightPanel = nullptr;
    LiveTcpServer* liveTcpServer;
    LiveHttpServer* liveHttpServer;
    LiveSignalsBridge* liveSignalsBridge;

    enum ActionType {
        //File
        ACTION_NEW = 0,
        ACTION_OPEN,
        ACTION_SAVE,
        ACTION_SAVE_AS,
        ACTION_IMPORT,
        //File export
        ACTION_EXPORT_GRAPH,
        ACTION_SCREEN_SHOOT,
        ACTION_RECORD_VIDEO,

        ACTION_CLOSE,
        //Edit
        ACTION_UNDO,
        ACTION_REDO,
        ACTION_COPY,
        ACTION_PASTE,
        ACTION_CUT,
        ACTION_COLLASPE,
        ACTION_EXPAND,
        ACTION_EASY_GRAPH,
        ACTION_OPEN_VIEW,
        ACTION_CLEAR_VIEW,
        //Render
        ACTION_SMOOTH_SHADING,
        ACTION_NORMAL_CHECK,
        ACTION_WIRE_FRAME,
        ACTION_SHOW_GRID,
        ACTION_BACKGROUND_COLOR,
        ACTION_SOLID,
        ACTION_SHADING,
        ACTION_OPTIX,
        //View EnvTex
        ACTION_BLACK_WHITE,
        ACTION_GREEK,
        ACTION_DAY_LIGHT,
        ACTION_DEFAULT,
        ACTION_FOOTBALL_FIELD,
        ACTION_FOREST,
        ACTION_LAKE,
        ACTION_SEA,
        //View Camera
        ACTION_NODE_CAMERA,
        //Window
        ACTION_SAVE_LAYOUT,
        ACTION_LAYOUT_MANAGE,
        //Help
        ACTION_LANGUAGE,
        ACTION_SHORTCUTLIST,
        //Others
        ACTION_CUSTOM_UI,
        ACTION_SET_NASLOC,
        ACTION_ZENCACHE,
        ACTION_ZOOM,
        ACTION_SELECT_NODE,
        ACTION_SNAPGRID,
        ACTION_SHOWGRID,
        ACTION_GROUP,
    };
signals:
    void recentFilesChanged();
    void visObjectsUpdated(ViewportWidget* viewport, int frameid);
    void visFrameUpdated(int);
    void alwaysModeChanged(bool bAlways);

public slots:
    void openFileDialog();
    void onNewFile();
    bool openFile(QString filePath);
    bool saveFile(QString filePath);
    bool saveQuit();
    void save();
    bool saveAs();
    void onMaximumTriggered();
    void onMenuActionTriggered(bool bTriggered);
    void onSplitDock(bool);
    void onCloseDock();
    void importGraph();
    void exportGraph();
    void onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    void onPrimitiveSelected(const std::unordered_set<std::string>& primids);
    void updateViewport(const QString& action = "");
    void onRunFinished();
    void onFeedBack();
    void clearErrorMark();
    void updateLightList();
    void saveDockLayout();
    void loadSavedLayout();
    void onLangChanged(bool bChecked);
    void directlyRunRecord(const ZENO_RECORD_RUN_INITPARAM& param);
    void onRunTriggered();
    void updateNativeWinTitle(const QString& title);
    void toggleTimelinePlay(bool bOn);

protected:
    void resizeEvent(QResizeEvent* event) override;
    bool event(QEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void init();
    void initMenu();
    void initLive();
    void initDocks();
    void initWindowProperty();
    void initDocksWidget(ZTabDockWidget* pCake, PtrLayoutNode root);
    void _resizeDocks(PtrLayoutNode root);
    void resetDocks(PtrLayoutNode root);
    void initTimelineDock();
    void recordRecentFile(const QString& filePath);
    void saveLayout2();
    void SplitDockWidget(ZTabDockWidget* after, ZTabDockWidget* dockwidget, Qt::Orientation orientation);
    QString getOpenFileByDialog();
    void setActionProperty();
    void screenShoot();
    void setActionIcon(QAction* action);
    void initCustomLayoutAction(const QStringList& list, bool isDefault = false);
    void loadDockLayout(QString name, bool isDefault = false);
    QJsonObject readDefaultLayout();
    void manageCustomLayout();
    void updateLatestLayout(const QString &layout);
    void loadRecentFiles();


    ZTimeline* m_pTimeline;
    PtrLayoutNode m_layoutRoot;
    bool m_bInDlgEventloop;
    bool m_bAlways;
    int m_nResizeTimes;
    Ui::MainWindow* m_ui;

    std::shared_ptr<ZCacheMgr> m_spCacheMgr;
};

#endif
