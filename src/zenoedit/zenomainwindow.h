#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <QtWidgets>

class ZenoDockWidget;
class ZenoGraphsEditor;

class ZenoMainWindow : public QMainWindow
{
    Q_OBJECT

    enum DOCK_TYPE
    {
        DOCK_VIEW,
        DOCK_EDITOR,
        DOCK_NODE_PARAMS,
        DOCK_NODE_DATA,
        DOCK_TIMER
    };

public:
    ZenoMainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ZenoMainWindow();

public slots:
    void onRunClicked(int nFrames);
    void openFileDialog();
    void openFile(QString filePath);
    void saveQuit();
    void saveAs();
    void onMaximumTriggered();
    void onSplitDock(bool);
    void onToggleDockWidget(DOCK_TYPE, bool);

private:
    void init();
    void initMenu();
    void initDocks();
    void houdiniStyleLayout();
    void simpleLayout();
    void simpleLayout2();
    void arrangeDocks2();
    void arrangeDocks3();
    void writeHoudiniStyleLayout();
    void writeSettings2();
    void readHoudiniStyleLayout();
    void readSettings2();
    QString getOpenFileByDialog();

    ZenoDockWidget *m_viewDock;
    ZenoDockWidget *m_editor;
    ZenoDockWidget *m_data;
    ZenoDockWidget *m_parameter;
    ZenoDockWidget *m_toolbar;
    ZenoDockWidget *m_shapeBar;
    ZenoDockWidget *m_timelineDock;

    //QVector<ZenoDockWidget*> m_docks;
    QMultiMap<DOCK_TYPE, ZenoDockWidget*> m_docks;

    ZenoGraphsEditor* m_pEditor;
};

#endif
