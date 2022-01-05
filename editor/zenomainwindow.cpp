#include "zenomainwindow.h"
#include <comctrl/zenodockwidget.h>
#include "nodesview/znodeseditwidget.h"
#include "panel/zenodatapanel.h"


ZenoMainWindow::ZenoMainWindow(QWidget *parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
{
    init();
}

void ZenoMainWindow::init()
{
    initMenu();
    initDocks();
}

void ZenoMainWindow::initDocks()
{
    m_view = new ZenoDockWidget("view", this);
    m_view->setObjectName(QString::fromUtf8("dock_view"));
    m_view->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    m_view->setWidget(new QWidget);
    addDockWidget(Qt::LeftDockWidgetArea, m_view);

    m_data = new ZenoDockWidget("data", this);
    m_data->setObjectName(QString::fromUtf8("dock_data"));
    m_data->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    ZenoDataPanel *pDataPanel = new ZenoDataPanel;
    m_data->setWidget(pDataPanel);
    addDockWidget(Qt::BottomDockWidgetArea, m_data);

    m_editor = new ZenoDockWidget("", this);
    m_editor->setObjectName(QString::fromUtf8("dock_editor"));
    m_editor->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    ZNodesEditWidget* nodesView = new ZNodesEditWidget;
    m_editor->setWidget(nodesView);
    addDockWidget(Qt::BottomDockWidgetArea, m_editor);

    m_parameter = new ZenoDockWidget("parameter", this);
    m_parameter->setObjectName(QString::fromUtf8("dock_parameter"));
    m_parameter->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable);
    m_parameter->setWidget(new QWidget);
    addDockWidget(Qt::RightDockWidgetArea, m_parameter);



    //tabifyDockWidget(m_parameter, m_data);
}

void ZenoMainWindow::initMenu()
{
    QMenuBar *pMenuBar = new QMenuBar(this);
    if (!pMenuBar)
        return;

    pMenuBar->setMaximumHeight(26);//todo: sizehint

    QMenu *pFile = new QMenu(tr("File"));
    {
        QAction *pAction = new QAction(tr("New"), pFile);
        pAction->setCheckable(false);
        pFile->addAction(pAction);

        pAction = new QAction(tr("Open"), pFile);
        pAction->setCheckable(false);
        pFile->addAction(pAction);

        pAction = new QAction(tr("Save"), pFile);
        pAction->setCheckable(false);
        pFile->addAction(pAction);

        pAction = new QAction(tr("Quit"), pFile);
        pAction->setCheckable(false);
        pFile->addAction(pAction);
    }

    QMenu *pEdit = new QMenu(tr("Edit"));
    {
        QAction *pAction = new QAction(tr("Undo"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);

        pAction = new QAction(tr("Redo"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);

        pAction = new QAction(tr("Cut"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);

        pAction = new QAction(tr("Copy"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);

        pAction = new QAction(tr("Paste"), pEdit);
        pAction->setCheckable(false);
        pEdit->addAction(pAction);
    }

    QMenu *pRender = new QMenu(tr("Render"));

    QMenu *pView = new QMenu(tr("View"));

    QMenu *pWindow = new QMenu(tr("Window"));

    QMenu *pHelp = new QMenu(tr("Help"));

    pMenuBar->addMenu(pFile);
    pMenuBar->addMenu(pEdit);
    pMenuBar->addMenu(pRender);
    pMenuBar->addMenu(pView);
    pMenuBar->addMenu(pWindow);
    pMenuBar->addMenu(pHelp);

    setMenuBar(pMenuBar);
}
