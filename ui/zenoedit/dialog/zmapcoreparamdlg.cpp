#include "zmapcoreparamdlg.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/iparammodel.h>
#include "ui_zmapcoreparamdlg.h"


struct TempItem : public QStandardItem
{
public:
    explicit TempItem(const QString& text) : QStandardItem(text) {
    }

    QPersistentModelIndex m_coreparam;
};


ZMapCoreparamDlg::ZMapCoreparamDlg(const QPersistentModelIndex& idx, QWidget* parent)
    : QDialog(parent)
    , m_model(nullptr)
{
    m_ui = new Ui::MapCoreparamDlg;
    m_ui->setupUi(this);

    m_ui->treeView->header()->setVisible(false);

    connect(m_ui->buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
    connect(m_ui->buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    if (pGraphsModel)
    {
        IParamModel* pInputs = pGraphsModel->paramModel(idx, PARAM_INPUT);
        m_model = new QStandardItemModel;
        m_ui->treeView->setModel(m_model);
        QStandardItem* pRoot = m_model->invisibleRootItem();
        if (pInputs)
        {
            int n = pInputs->rowCount();
            if (n > 0)
            {
                QStandardItem* pGroup = new QStandardItem("Input Sockets:");
                pGroup->setSelectable(false);
                pRoot->appendRow(pGroup);
                for (int r = 0; r < n; r++)
                {
                    const QModelIndex& idx = pInputs->index(r, 0);
                    const QString& name = idx.data(ROLE_PARAM_NAME).toString();
                    TempItem* pItem = new TempItem(name);
                    pItem->m_coreparam = idx;
                    pGroup->appendRow(pItem);
                }
            }
        }

        IParamModel* pParams = pGraphsModel->paramModel(idx, PARAM_PARAM);
        if (pParams)
        {
            int n = pParams->rowCount();
            if (n > 0)
            {
                QStandardItem* pGroup = new QStandardItem("Parameters:");
                pGroup->setSelectable(false);
                pRoot->appendRow(pGroup);
                for (int r = 0; r < n; r++)
                {
                    const QModelIndex& idx = pParams->index(r, 0);
                    const QString& name = idx.data(ROLE_PARAM_NAME).toString();
                    TempItem* pItem = new TempItem(name);
                    pItem->m_coreparam = idx;
                    pGroup->appendRow(pItem);
                }
            }
        }

        IParamModel* pOutputs = pGraphsModel->paramModel(idx, PARAM_OUTPUT);
        if (pOutputs)
        {
            int n = pOutputs->rowCount();
            if (n > 0)
            {
                QStandardItem* pGroup = new QStandardItem("Output Sockets:");
                pGroup->setSelectable(false);
                pRoot->appendRow(pGroup);
                for (int r = 0; r < n; r++)
                {
                    const QModelIndex& idx = pOutputs->index(r, 0);
                    const QString& name = idx.data(ROLE_PARAM_NAME).toString();
                    TempItem* pItem = new TempItem(name);
                    pItem->m_coreparam = idx;
                    pGroup->appendRow(pItem);
                }
            }
        }

        TempItem* pNullItem = new TempItem("Null Item");
        pRoot->appendRow(pNullItem);
    }
    m_ui->treeView->expandAll();
}

QModelIndex ZMapCoreparamDlg::coreIndex() const
{
    QStandardItem* pItem = m_model->itemFromIndex(m_ui->treeView->currentIndex());
    if (!pItem) return QModelIndex();

    TempItem* pTempItem = static_cast<TempItem*>(pItem);
    QString name = pTempItem->m_coreparam.data(ROLE_PARAM_NAME).toString();
    return pTempItem->m_coreparam;
}
