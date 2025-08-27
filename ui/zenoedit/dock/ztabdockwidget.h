#ifndef __ZTAB_DOCKWIDGET_H__
#define __ZTAB_DOCKWIDGET_H__

#include <unordered_set>
#include <QtWidgets>
#include "zenodockwidget.h"
#include "docktabcontent.h"
#include <zenoio/include/common.h>

class ZenoMainWindow;
class DisplayWidget;
class ZenoGraphsEditor;
class ZDockTabWidget;   //may confuse with ZTabDockWidget...

enum PANEL_TYPE
{
    PANEL_EMPTY,
    PANEL_GL_VIEW,
    PANEL_EDITOR,
    PANEL_NODE_PARAMS,
    PANEL_NODE_DATA,
    PANEL_LOG,
    PANEL_LIGHT,
    PANEL_IMAGE,
    PANEL_OPTIX_VIEW,
    PANEL_COMMAND_PARAMS,
    PANEL_OPEN_PATH,
    PANEL_OUTLINE,
    PANEL_XFORM
};

class ZTabDockWidget : public QDockWidget
{
    Q_OBJECT
    typedef QDockWidget _base;
public:
    explicit ZTabDockWidget(ZenoMainWindow* parent, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ZTabDockWidget();

    int count() const;
    QWidget* widget(int i) const;
    QWidget* widget() const;
    DisplayWidget* getUniqueViewport() const;
    QVector<DisplayWidget*> viewports() const;
    ZenoGraphsEditor* getAnyEditor() const;
    void setCurrentWidget(PANEL_TYPE type);
    void onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    void onPrimitiveSelected(const std::unordered_set<std::string>& primids, std::string mtlid = "", bool selecFromOpitx = false);
    void onUpdateViewport(const QString& action);
    void updateLights();
    void cleanupView();

    static PANEL_TYPE title2Type(const QString &title);

public slots:
    void onPlayClicked(bool);
    void onSliderValueChanged(int);
    void onRunFinished();
    void onAddTab(PANEL_TYPE type);
    void onAddTab(PANEL_TYPE type, DockContentWidgetInfo info);
    void onMenuActionTriggered(QAction* pAction, bool bTriggered);

protected:
    void paintEvent(QPaintEvent* event) override;
    bool event(QEvent* event) override;

signals:
    void maximizeTriggered();
    void floatTriggered();
    void splitRequest(bool bHorzonal);
    void closeRequest();
    void nodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes);

private slots:
    void onDockOptionsClicked();
    void onMaximizeTriggered();
    void onFloatTriggered();
    void onAddTabClicked();

private:
    void init(ZenoMainWindow* pMainWin);
    bool isTopLevelWin();
    QWidget* createTabWidget(PANEL_TYPE type);
    QString type2Title(PANEL_TYPE type);


    PANEL_TYPE m_debugPanel;

    Qt::WindowFlags m_oldFlags;
    Qt::WindowFlags m_newFlags;
    ZDockTabWidget* m_tabWidget;
};

#endif