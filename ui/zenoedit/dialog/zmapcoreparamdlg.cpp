#include "zmapcoreparamdlg.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/nodeparammodel.h>
#include "ui_zmapcoreparamdlg.h"
#include "variantptr.h"


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
        NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(idx.data(ROLE_NODE_PARAMS));
        m_model = nodeParams;
        m_ui->treeView->setModel(nodeParams);
    }
    m_ui->treeView->expandAll();
}

QModelIndex ZMapCoreparamDlg::coreIndex() const
{
    QStandardItem* pItem = m_model->itemFromIndex(m_ui->treeView->currentIndex());
    if (!pItem)
        return QModelIndex();
    return pItem->index();
}
