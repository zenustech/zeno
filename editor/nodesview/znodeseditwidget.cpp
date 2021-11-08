#include "znodeseditwidget.h"
#include "znodeswebview.h"
#include "znodesgraphicsview.h"
#include <QMenuBar>


ZNodesEditWidget::ZNodesEditWidget(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* pLayout = new QVBoxLayout;

    QMenuBar* pMenu = new QMenuBar;
	pMenu->setMinimumHeight(26);
	initMenu(pMenu);
    pLayout->addWidget(pMenu);

    QPushButton* pBtn = new QPushButton;
    pBtn->setText(tr("reload"));
    QHBoxLayout* pHLayout = new QHBoxLayout;
    pHLayout->addWidget(pBtn);
    pHLayout->addStretch();

    pLayout->addLayout(pHLayout);

    ZNodesWebEngineView* pView = new ZNodesWebEngineView;
	QTabWidget* pTab = new QTabWidget;
	pTab->addTab(pView, "webview");

	ZNodesGraphicsView* pGraphicsView = new ZNodesGraphicsView;
	pTab->addTab(pGraphicsView, "native");
    pLayout->addWidget(pTab);

    setLayout(pLayout);

    connect(pBtn, SIGNAL(clicked()), pView, SLOT(reload()));
}

void ZNodesEditWidget::initMenu(QMenuBar* pMenu)
{
    QMenu* pFile = new QMenu(tr("File"));
	{
		QAction* pAction = new QAction(tr("New"), pFile);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+N")));
		pFile->addAction(pAction);

		pAction = new QAction(tr("Open"), pFile);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+O")));
		pFile->addAction(pAction);

		pAction = new QAction(tr("Save"), pFile);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+S")));
		pFile->addAction(pAction);

		pAction = new QAction(tr("Export"), pFile);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+Shift+E")));
		pFile->addAction(pAction);
	}

    QMenu* pEdit = new QMenu(tr("Edit"));
	{
		QAction* pAction = new QAction(tr("Undo"), pEdit);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+Z")));
		pEdit->addAction(pAction);

		pAction = new QAction(tr("Redo"), pEdit);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+Y")));
		pEdit->addAction(pAction);

		pAction = new QAction(tr("Copy"), pEdit);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+C")));
		pEdit->addAction(pAction);

		pAction = new QAction(tr("Paste"), pEdit);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+V")));
		pEdit->addAction(pAction);

        pAction = new QAction(tr("Find"), pEdit);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+F")));
		pEdit->addAction(pAction);
	}

    pMenu->addMenu(pFile);
    pMenu->addMenu(pEdit);
}