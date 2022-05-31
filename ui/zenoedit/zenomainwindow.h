#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <QtWidgets>
#include "dock/zenodockwidget.h"

class ZenoDockWidget;
class ZenoGraphsEditor;

class ZenoMainWindow : public QMainWindow
{
    Q_OBJECT
public:
    ZenoMainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ZenoMainWindow();
    ZenoGraphsEditor* editor() const { return m_pEditor; }
    bool inDlgEventLoop() const;
    void setInDlgEventLoop(bool bOn);

public slots:
    void onRunClicked(int beginFrame, int endFrame);
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
    void updateViewport(const QString& action = "");
    void onFeedBack();

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void init();
    void initMenu();
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
