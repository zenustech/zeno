#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <unordered_set>
#include <QtWidgets>
#include "dock/ztabdockwidget.h"
#include "panel/zenolights.h"
#include "common.h"
#include "layout/winlayoutrw.h"


class DisplayWidget;
class ZTimeline;

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
    void resetTimeline(TIMELINE_INFO info);
    ZTimeline* timeline() const;
    DisplayWidget *getDisplayWidget();
    void dispatchCommand(QAction* pAction, bool bTriggered);

    QLineEdit* selected = nullptr;
    ZenoLights* lightPanel = nullptr;
    float mouseSen = 0.2;

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
        ACTION_SHADONG,
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
        //Window Custom Layout
        ACTION_DEFAULT_LAYOUT,
        //Help
        ACTION_LANGUAGE,
        //Others
        ACTION_CUSTOM_UI,
        ACTION_SET_NASLOC,
        ACTION_ZENCACHE,
        ACTION_ZOOM,
        ACTION_SELECT_NODE
    };

public slots:
    void openFileDialog();
    void onNewFile();
    bool openFile(QString filePath);
    bool saveFile(QString filePath);
    bool saveQuit();
    void save();
    void saveAs();
    void onMaximumTriggered();
    void onMenuActionTriggered(bool bTriggered);
    void onSplitDock(bool);
    void onCloseDock();
    void onToggleDockWidget(DOCK_TYPE, bool);
    void onDockSwitched(DOCK_TYPE);
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

protected:
    void resizeEvent(QResizeEvent* event) override;
    bool event(QEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void init();
    void initMenu();
    void initDocks();
    void initDocksWidget(ZTabDockWidget* pCake, PtrLayoutNode root);
    void _resizeDocks(PtrLayoutNode root);
    void resetDocks(PtrLayoutNode root);
    void initTimelineDock();
    void recordRecentFile(const QString& filePath);
    void saveLayout2();
    void SplitDockWidget(ZTabDockWidget* after, ZTabDockWidget* dockwidget, Qt::Orientation orientation);
    QString getOpenFileByDialog();
    QString uniqueDockObjName(DOCK_TYPE type);
    void setActionProperty();
    void screenShoot();
    void setActionIcon(QAction *action);

    ZTimeline* m_pTimeline;
    PtrLayoutNode m_layoutRoot;
    bool m_bInDlgEventloop;
    int m_nResizeTimes;
    Ui::MainWindow* m_ui;
};

#endif
