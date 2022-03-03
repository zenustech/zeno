#ifndef __ZENO_MAINWINDOW_H__
#define __ZENO_MAINWINDOW_H__

#include <QtWidgets>

enum DOCK_TYPE
{
	DOCK_VIEW,
	DOCK_EDITOR,
	DOCK_NODE_PARAMS,
	DOCK_NODE_DATA,
	DOCK_TIMER
};

class ZenoDockWidget;
class ZenoGraphsEditor;

class ZenoMainWindow : public QMainWindow
{
    Q_OBJECT
public:
    ZenoMainWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ZenoMainWindow();
    ZenoGraphsEditor* editor() const { return m_pEditor; };

public slots:
    void onRunClicked(int nFrames);
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

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void init();
    void initMenu();
    void initDocks();
    void houdiniStyleLayout();
    void verticalLayout();
    void onlyEditorLayout();
    void simpleLayout2();
    void arrangeDocks2();
    void arrangeDocks3();
    void writeHoudiniStyleLayout();
    void writeSettings2();
    void readHoudiniStyleLayout();
    void readSettings2();
    void adjustDockSize();
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
    QString currFilePath;//TODO:luzh fix me

    ZenoGraphsEditor* m_pEditor;
};

#endif
