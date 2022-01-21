#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <QtWidgets>

class ZenoDockWidget;

class ZenoMainWindow : public QMainWindow
{
    Q_OBJECT
public:
    ZenoMainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());

signals:
    void modelInited();

public slots:
    void onRunClicked(int nFrames);
    void openFileDialog();
    void saveQuit();
    void saveAs();

private:
    void init();
    void initMenu();
    void initDocks();
    void houdiniStyleLayout();
    void simplifyLayout();
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
};

#endif