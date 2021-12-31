#include "znodeseditwidget.h"
#include "znodeswebview.h"
#include "znodesgraphicsview.h"
#include <util/uihelper.h>
#include <nodesys/zenographswidget.h>
#include <io/zsgreader.h>
#include <io/zsgwriter.h>
#include <model/graphsmodel.h>
#include <QMenuBar>


ZNodesEditWidget::ZNodesEditWidget(QWidget* parent)
    : QWidget(parent)
    , m_pGraphsWidget(nullptr)
    , m_pComboSubGraph(nullptr)
    , m_pReloadBtn(nullptr)
    , m_pDeleteBtn(nullptr)
    , m_pNewSubGraph(nullptr)
    , m_model(nullptr)
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

        m_pReloadBtn = new QPushButton("Reload");
        pHLayout->addWidget(m_pReloadBtn);

		m_pDeleteBtn = new QPushButton("Delete");
        pHLayout->addWidget(m_pDeleteBtn);
		pHLayout->addStretch();
	}
    pLayout->addLayout(pHLayout);

	m_pGraphsWidget = new ZenoGraphsWidget;
    pLayout->addWidget(m_pGraphsWidget);

    setLayout(pLayout);
}

QString ZNodesEditWidget::getOpenFileByDialog()
{
    const QString &initialPath = ".";
    QFileDialog fileDialog(this, tr("Open"), initialPath, "Zensim Graph File (*.zsg)\nAll Files (*)");
    fileDialog.setAcceptMode(QFileDialog::AcceptOpen);
    fileDialog.setFileMode(QFileDialog::ExistingFile);
    fileDialog.setDirectory(initialPath);
    if (fileDialog.exec() != QDialog::Accepted)
        return "";

    QString filePath = fileDialog.selectedFiles().first();
    return filePath;
}

void ZNodesEditWidget::openFileDialog()
{
    QString filePath = getOpenFileByDialog();
    if (filePath.isEmpty())
        return;

    if (m_model != nullptr)
    {
        //todo: save first
        m_model = nullptr;
    }

    m_model = ZsgReader::getInstance().loadZsgFile(filePath);
    m_model->initDescriptors();

    m_pGraphsWidget->setGraphsModel(m_model);
    m_pComboSubGraph->setModel(m_model);

    connect(m_pReloadBtn, &QPushButton::clicked, [=]() {
        m_model->reloadSubGraph(m_pComboSubGraph->currentText());
    });
    connect(m_pDeleteBtn, SIGNAL(clicked()), m_model, SLOT(onRemoveCurrentItem()));

    connect(m_pComboSubGraph, SIGNAL(currentIndexChanged(int)), m_model, SLOT(onCurrentIndexChanged(int)));
    connect(m_model->selectionModel(), &QItemSelectionModel::currentChanged, 
        [=](const QModelIndex &current, const QModelIndex &previous) {
            m_pGraphsWidget->setCurrentIndex(current.row());
            m_pComboSubGraph->setCurrentIndex(current.row());
    });

    //menu
    connect(m_pNewSubGraph, SIGNAL(triggered()), this, SLOT(onSubGraphTriggered()));
}

void ZNodesEditWidget::importGraph()
{
    QString filePath = getOpenFileByDialog();
    if (filePath.isEmpty())
        return;

    GraphsModel* pModel = ZsgReader::getInstance().loadZsgFile(filePath);
    m_model->setDescriptors(pModel->descriptors());
    for (int i = 0; i < pModel->rowCount(); i++)
    {
        SubGraphModel *pSubGraphModel = pModel->subGraph(i);
        QString name = pSubGraphModel->name();
        if (SubGraphModel* pExist = m_model->subGraph(name))
        {
            //todo: reload
        }
        else
        {
            m_model->appendSubGraph(pSubGraphModel->clone(m_model));
        }
    }
    m_model->switchSubGraph("main");
    m_model->initDescriptors();
}

void ZNodesEditWidget::exportGraph()
{
    QString path = QFileDialog::getSaveFileName(this, "Path to Export", ""
        , "C++ Source File(*.cpp);; C++ Header File(*.h);; JSON file(*.json);; All Files(*);;"
        , nullptr
        , QFileDialog::DontConfirmOverwrite);
    if (!path.isEmpty())
    {

    }
}

void ZNodesEditWidget::saveAs()
{
    QString path = QFileDialog::getSaveFileName(this, "Path to Save", "", "Zensim Graph File(*.zsg);; All Files(*);;");
    if (!path.isEmpty())
    {
        QString strContent = ZsgWriter::getInstance().dumpProgram(m_model);
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly)) {
            qWarning() << Q_FUNC_INFO << "Failed to open" << path << f.errorString();
            return;
        }
        f.write(strContent.toUtf8());
        f.close();
    }
}

void ZNodesEditWidget::onSubGraphTriggered()
{
    QDialog dialog(this);
    QVBoxLayout* pLayout = new QVBoxLayout;
    QLineEdit* pLineEdit = new QLineEdit;
    pLayout->addWidget(pLineEdit);

    QHBoxLayout* pHLayout = new QHBoxLayout;
    pHLayout->addStretch();
    QPushButton* pOk = new QPushButton("OK");
    QPushButton* pCancel = new QPushButton("Cancel");
    pHLayout->addWidget(pOk);
    pHLayout->addWidget(pCancel);
    pLayout->addLayout(pHLayout);

    connect(pOk, SIGNAL(clicked()), &dialog, SLOT(accept()));
    connect(pCancel, SIGNAL(clicked()), &dialog, SLOT(reject()));

    dialog.setLayout(pLayout);
    QButtonGroup* pBtnGroup = new QButtonGroup;
    if (dialog.exec() == QDialog::Accepted)
    {
        GraphsModel* pModel = m_pGraphsWidget->model();
        pModel->newSubgraph(pLineEdit->text());
    }
}

void ZNodesEditWidget::initMenu(QMenuBar* pMenu)
{
    QMenu* pFile = new QMenu(tr("File"));
	{
		QAction* pAction = new QAction(tr("New"), pFile);
        QMenu *pNewMenu = new QMenu;
        QAction* pNewGraph = pNewMenu->addAction("New Graph");
        m_pNewSubGraph = pNewMenu->addAction("New Subgraph");
        
        pAction->setMenu(pNewMenu);

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

        pAction = new QAction(tr("Save As"), pFile);
        pAction->setCheckable(false);
        connect(pAction, SIGNAL(triggered()), this, SLOT(saveAs()));
        pFile->addAction(pAction);

        pAction = new QAction(tr("Import"), pFile);
        pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+Shift+O")));
        connect(pAction, SIGNAL(triggered()), this, SLOT(importGraph()));
        pFile->addAction(pAction);

		pAction = new QAction(tr("Export"), pFile);
		pAction->setCheckable(false);
        pAction->setShortcut(QKeySequence(tr("Ctrl+Shift+E")));
        connect(pAction, SIGNAL(triggered()), this, SLOT(exportGraph()));
		pFile->addAction(pAction);
	}

    QMenu* pEdit = new QMenu(tr("Edit"));
	{
		QAction* pAction = new QAction(tr("Undo"), pEdit);
		pAction->setCheckable(false);
		pEdit->addAction(pAction);

		pAction = new QAction(tr("Redo"), pEdit);
		pAction->setCheckable(false);
		pEdit->addAction(pAction);

		pAction = new QAction(tr("Copy"), pEdit);
		pAction->setCheckable(false);
		pEdit->addAction(pAction);

		pAction = new QAction(tr("Paste"), pEdit);
		pAction->setCheckable(false);
		pEdit->addAction(pAction);

        pAction = new QAction(tr("Find"), pEdit);
		pAction->setCheckable(false);
		pEdit->addAction(pAction);
	}

    pMenu->addMenu(pFile);
    pMenu->addMenu(pEdit);
}