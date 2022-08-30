#ifndef __ZTAB_DOCKWIDGET_H__
#define __ZTAB_DOCKWIDGET_H__

#include <unordered_set>
#include <QtWidgets>
#include "zenodockwidget.h"

class ZenoMainWindow;

class ZDockTabWidget;   //may confuse with ZTabDockWidget...

enum PANEL_TYPE
{
    PANEL_EMPTY,
    PANEL_VIEW,
    PANEL_EDITOR,
    PANEL_NODE_PARAMS,
    PANEL_NODE_DATA,
    PANEL_LOG,
};

class ZTabDockWidget : public QDockWidget
{
    Q_OBJECT
    typedef QDockWidget _base;
public:
    explicit ZTabDockWidget(ZenoMainWindow* parent, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ZTabDockWidget();

    void setCurrentWidget(PANEL_TYPE type);
    void onNodesSelected(const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    void onPrimitiveSelected(const std::unordered_set<std::string>& primids);
    void onUpdateViewport(const QString& action);
    void onRunFinished();

protected:
    void paintEvent(QPaintEvent* event) override;
    bool event(QEvent* event) override;

signals:
    void maximizeTriggered();
    void floatTriggered();
    void splitRequest(bool bHorzonal);
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
    PANEL_TYPE title2Type(const QString& title);

    Qt::WindowFlags m_oldFlags;
    Qt::WindowFlags m_newFlags;
    ZDockTabWidget* m_tabWidget;
    QLabel* m_plblName;
    QLineEdit* m_pLineEdit;
};

#endif