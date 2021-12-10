#include "znodeseditwidget.h"
#include "znodeswebview.h"
#include "znodesgraphicsview.h"
#include <nodesys/zenographswidget.h>
#include <io/zsgreader.h>
#include <model/graphsmodel.h>
#include <QMenuBar>


ZNodesEditWidget::ZNodesEditWidget(QWidget* parent)
    : QWidget(parent)
    , m_pGraphsWidget(nullptr)
    , m_pComboSubGraph(nullptr)
    , m_pNewBtn(nullptr)
    , m_pDeleteBtn(nullptr)
{
    QVBoxLayout *pLayout = new QVBoxLayout;

    QMenuBar* pMenu = new QMenuBar;
	pMenu->setMinimumHeight(26);
	initMenu(pMenu);
    pLayout->addWidget(pMenu);

    QHBoxLayout* pHLayout = new QHBoxLayout;
	{
        m_pComboSubGraph = new QComboBox;
        pHLayout->addWidget(m_pComboSubGraph);

        m_pNewBtn = new QPushButton("New");
        pHLayout->addWidget(m_pNewBtn);

		m_pDeleteBtn = new QPushButton("Delete");
        pHLayout->addWidget(m_pDeleteBtn);
		pHLayout->addStretch();
	}
    pLayout->addLayout(pHLayout);

	m_pGraphsWidget = new ZenoGraphsWidget;
    pLayout->addWidget(m_pGraphsWidget);

    setLayout(pLayout);
}

void ZNodesEditWidget::openFileDialog()
{
    const QString &initialPath = ".";
    QFileDialog fileDialog(this, tr("Open"), initialPath, "Zensim Graph File (*.zsg)\nAll Files (*)");
    fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
    fileDialog.setFileMode(QFileDialog::ExistingFile);
    fileDialog.setDirectory(initialPath);
    if (fileDialog.exec() != QDialog::Accepted)
        return;

    QString filePath = fileDialog.selectedFiles().first();
    GraphsModel *pModel = ZsgReader::getInstance().loadZsgFile(filePath);
    m_pGraphsWidget->setGraphsModel(pModel);
    m_pComboSubGraph->setModel(pModel);

    connect(m_pComboSubGraph, SIGNAL(currentIndexChanged(int)), pModel, SLOT(onCurrentIndexChanged(int)));
    connect(pModel, SIGNAL(itemSelected(int)), m_pComboSubGraph, SLOT(setCurrentIndex(int)));
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
        connect(pAction, SIGNAL(triggered()), this, SLOT(openFileDialog()));
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