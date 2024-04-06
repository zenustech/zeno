#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <unordered_set>
#include <QtWidgets>
#include "layout/zdockwidget.h"
#include "panel/zenolights.h"
#include "layout/winlayoutrw.h"
#include <QTcpSocket>
#include <QLocalSocket>
#include "DockManager.h"


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
    ZenoMainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags(), PANEL_TYPE onlyView = PANEL_EMPTY);
    ~ZenoMainWindow();
    bool inDlgEventLoop() const;
    void setInDlgEventLoop(bool bOn);
    zeno::TimelineInfo timelineInfo();
    void setAlways(bool bAlways);
    void setAlwaysLightCameraMaterial(bool bAlwaysLightCamera, bool bAlwaysMaterial);
    bool isAlways() const;
    bool isAlwaysLightCamera() const;
    bool isAlwaysMaterial() const;
    void resetTimeline(zeno::TimelineInfo info);
    void initUserdata(USERDATA_SETTING info);
    ZTimeline* timeline() const;
    QVector<DisplayWidget*> viewports() const;
    DisplayWidget* getCurrentViewport() const;
    DisplayWidget* getOptixWidget() const;
    ZenoGraphsEditor* getAnyEditor() const;
    void dispatchCommand(QAction* pAction, bool bTriggered);

    void doFrameUpdate(int frame);
    void sortRecentFile(QStringList &lst);
    bool isOnlyOptixWindow() const;
    bool isRecordByCommandLine() const;

    QLineEdit* selected = nullptr;
    ZenoLights* lightPanel = nullptr;

    enum ActionType {
        //File
        ACTION_NEWFILE = 0,
        ACTION_NEW_SUBGRAPH,
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
        ACTION_UV_MODE,
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
        ACTION_NODE_EDITOR,
        ACTION_GL_VIEWPORT,
        ACTION_NODE_PARAMETERS,
        ACTION_OBJECT_DATA,
        ACTION_LOG,
        ACTION_OPTIX_VIEW,
        ACTION_IMAGE,
        ACTION_COMMAND_ARGS,
        ACTION_OPEN_PATH,
        ACTION_SAVE_LAYOUT,
        ACTION_LAYOUT_MANAGE,
        //Help
        ACTION_LANGUAGE,
        ACTION_SHORTCUTLIST,
        ACTION_FEEDBACK,
        ACTION_ABOUT,
        ACTION_CHECKUPDATE,
        //options
        ACTION_SET_NASLOC,
        ACTION_ZENCACHE,
        ACTION_SET_SHORTCUT,
        ACTION_PREFERENCES,
        //Others
        ACTION_CUSTOM_UI,
        ACTION_ZOOM,
        ACTION_SELECT_NODE,
        ACTION_SNAPGRID,
        ACTION_SHOWGRID,
        ACTION_GROUP,
        ACTION_SHOWTHUMB,
        ACTION_REARRANGE_GRAPH,

    };
signals:
    void recentFilesChanged(const QObject *sender);
    void visObjectsUpdated(ViewportWidget* viewport, int frameid);
    void visFrameUpdated(bool bGLView, int frameid);
    void alwaysModeChanged(bool bAlways);
    void runFinished();

public slots:
    void openFileDialog();
    void onNewFile();
    bool openFile(QString filePath);
    bool saveFile(QString filePath);
    bool saveQuit();
    void saveQuitShowWelcom();
    void save();
    bool saveAs();
    void onMaximumTriggered();
    void onMenuActionTriggered(bool bTriggered);
    void onSplitDock(bool);
    void onCloseDock();
    void importGraph(bool bPreset = false);
    void exportGraph();
    void onNodesSelected(GraphModel* subgraph, const QModelIndexList& nodes, bool select);
    void onPrimitiveSelected(const std::unordered_set<std::string>& primids);
    void updateViewport(const QString& action = "");
    void onRunFinished();
    void onFeedBack();
    void clearErrorMark();
    void updateLightList();
    void saveDockLayout();
    void loadSavedLayout();
    void onLangChanged(bool bChecked);
    void optixClientSend(QString& info);
    void optixClientStartRec();
    void onRunTriggered(bool applyLightAndCameraOnly = false, bool applyMaterialOnly = false);
    void updateNativeWinTitle(const QString& title);
    void toggleTimelinePlay(bool bOn);
    void onZenovisFrameUpdate(bool bGLView, int frameid);
    void onCheckUpdate();
    void onFrameSwitched(int frameid);
    void onCalcFinished(bool bSucceed, zeno::ObjPath nodeUuidPath, QString msg);

protected:
    void resizeEvent(QResizeEvent* event) override;
    bool event(QEvent* event) override;
    void closeEvent(QCloseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private:
    void init(PANEL_TYPE onlyView);
    void initMenu();
    void initDocks(PANEL_TYPE onlyView);
    void initAllDockWidgets();
    void initWindowProperty();
    void initDocksWidget(ads::CDockAreaWidget* cakeArea, ads::CDockWidget* pCake, PtrLayoutNode root);
    void _resizeDocks(PtrLayoutNode root);
    void resetDocks(PtrLayoutNode root);
    void initTimelineDock();
    void recordRecentFile(const QString& filePath);
    void saveLayout2();
    void SplitDockWidget(ZDockWidget* after, ZDockWidget* dockwidget, Qt::Orientation orientation);
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
    void initShortCut();
    void updateShortCut(QStringList keys);
    void shortCutDlg();
    void killOptix();
    DisplayWidget* getOnlyViewport() const;
    bool resetProc();
    void onCreatePanel(int actionType);

    ZTimeline* m_pTimeline;
    PtrLayoutNode m_layoutRoot;
    bool m_bInDlgEventloop;
    bool m_bAlways;
    bool m_bAlwaysLightCamera;
    bool m_bAlwaysMaterial;
    int m_nResizeTimes;
    bool m_bOnlyOptix;          //isolate optix window.
    Ui::MainWindow* m_ui;

    std::unique_ptr<QLocalSocket> optixClientSocket;
    bool m_bOptixProcRecording = false;

    bool m_bRecordByCommandLine = false;

    ads::CDockManager* m_pDockManager;
};

#endif
