#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <QtWidgets>

class ZenoDockWidget;

class ZenoMainWindow : public QMainWindow
{
    Q_OBJECT
public:
    ZenoMainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());

public slots:
    void onRunClicked(int nFrames);

private:
    void init();
    void initMenu();
    void initDocks();
    void houdiniStyleLayout();
    void arrangeDocks2();
    void arrangeDocks3();
    void writeHoudiniStyleLayout();
    void writeSettings2();
    void readHoudiniStyleLayout();
    void readSettings2();

    ZenoDockWidget *m_viewDock;
    ZenoDockWidget *m_editor;
    ZenoDockWidget *m_data;
    ZenoDockWidget *m_parameter;
    ZenoDockWidget *m_toolbar;
    ZenoDockWidget *m_shapeBar;
    ZenoDockWidget *m_timelineDock;
};

#endif