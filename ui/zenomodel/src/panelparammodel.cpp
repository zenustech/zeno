#include "panelparammodel.h"
#include "nodeparammodel.h"
#include "vparamitem.h"
#include "modelrole.h"
#include "iotags.h"


PanelParamModel::PanelParamModel(
            NodeParamModel* nodeParams,
            VPARAM_INFO root,
            const QModelIndex& nodeIdx,
            IGraphsModel *pModel,
            QObject *parent)
    : ViewParamModel(false, nodeIdx, pModel, parent)
{
    if (!root.children.isEmpty())
        importParamInfo(root);
    else
        initParams(nodeParams);
    connect(nodeParams, &NodeParamModel::rowsInserted, this, &PanelParamModel::onNodeParamsInserted);
    connect(nodeParams, &NodeParamModel::rowsAboutToBeRemoved, this, &PanelParamModel::onNodeParamsAboutToBeRemoved);
}

PanelParamModel::PanelParamModel(const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent)
    : ViewParamModel(false, nodeIdx, pModel, parent)
{
}

PanelParamModel::~PanelParamModel()
{
}

void PanelParamModel::initParams(NodeParamModel* nodeParams)
{
    ZASSERT_EXIT(nodeParams);
    auto root = nodeParams->invisibleRootItem();
    /*default structure:
                root
                    |-- Tab (Default)
                        |-- Inputs (Group)
                            -- input param1 (Item)
                            -- input param2
                            ...

                        |-- Params (Group)
                            -- param1 (Item)
                            -- param2 (Item)
                            ...

                        |- Outputs (Group)
                            - output param1 (Item)
                            - output param2 (Item)
                ...
            */
    VParamItem *pRoot = new VParamItem(VPARAM_ROOT, iotags::params::panel_root);
    pRoot->setEditable(false);

    VParamItem *pTab = new VParamItem(VPARAM_TAB, iotags::params::panel_default_tab);
    {
        VParamItem *pInputsGroup = new VParamItem(VPARAM_GROUP, iotags::params::panel_inputs);
        VParamItem *paramsGroup = new VParamItem(VPARAM_GROUP, iotags::params::panel_params);
        VParamItem *pOutputsGroup = new VParamItem(VPARAM_GROUP, iotags::params::panel_outputs);

        pInputsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);
        paramsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);
        pOutputsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);

        const VParamItem* pNodeInputs = nodeParams->getInputs();
        for (int r = 0; r < pNodeInputs->rowCount(); r++)
        {
            VParamItem* pNodeParam = static_cast<VParamItem*>(pNodeInputs->child(r));
            VParamItem* panelParam = static_cast<VParamItem*>(pNodeParam->clone());
            panelParam->mapCoreParam(pNodeParam->index());
            pInputsGroup->appendRow(panelParam);
        }

        const VParamItem* pNodeParams = nodeParams->getParams();
        for (int r = 0; r < pNodeParams->rowCount(); r++)
        {
            VParamItem* pNodeParam = static_cast<VParamItem*>(pNodeParams->child(r));
            VParamItem* panelParam = static_cast<VParamItem*>(pNodeParam->clone());
            panelParam->mapCoreParam(pNodeParam->index());
            paramsGroup->appendRow(panelParam);
        }

        const VParamItem* pNodeOutputs = nodeParams->getOutputs();
        for (int r = 0; r < pNodeOutputs->rowCount(); r++)
        {
            VParamItem* pNodeParam = static_cast<VParamItem*>(pNodeOutputs->child(r));
            VParamItem* panelParam = static_cast<VParamItem*>(pNodeParam->clone());
            panelParam->mapCoreParam(pNodeParam->index());
            pOutputsGroup->appendRow(panelParam);
        }

        pTab->appendRow(pInputsGroup);
        pTab->appendRow(paramsGroup);
        pTab->appendRow(pOutputsGroup);
    }
    pTab->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);

    pRoot->appendRow(pTab);
    appendRow(pRoot);
}

void PanelParamModel::onNodeParamsInserted(const QModelIndex &parent, int first, int last)
{
    QStandardItemModel* pModel = qobject_cast<QStandardItemModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex& idxNodeParam = pModel->index(first, 0, parent);
    if (!idxNodeParam.isValid())
        return;

    VParamItem* pNodeParam = static_cast<VParamItem*>(pModel->itemFromIndex(idxNodeParam));
    VParamItem* parentItem = static_cast<VParamItem*>(pNodeParam->parent());
    const QString& parentName = parentItem->m_name;
    if (parentName == iotags::params::node_inputs)
    {
        QList<QStandardItem*> lst = findItems(iotags::params::panel_inputs, Qt::MatchRecursive | Qt::MatchExactly);
        ZASSERT_EXIT(lst.size() == 1);
        VParamItem* pNewItem = static_cast<VParamItem*>(pNodeParam->clone());
        pNewItem->mapCoreParam(pNodeParam->index());
        lst[0]->appendRow(pNewItem);
    }
    else if (parentName == iotags::params::node_params)
    {
        QList<QStandardItem*> lst = findItems(iotags::params::panel_params, Qt::MatchRecursive | Qt::MatchExactly);
        ZASSERT_EXIT(lst.size() == 1);
        VParamItem* pNewItem = static_cast<VParamItem*>(pNodeParam->clone());
        pNewItem->mapCoreParam(pNodeParam->index());
        lst[0]->appendRow(pNewItem);
    }
    else if (parentName == iotags::params::node_outputs)
    {
        QList<QStandardItem*> lst = findItems(iotags::params::panel_outputs, Qt::MatchRecursive | Qt::MatchExactly);
        ZASSERT_EXIT(lst.size() == 1);
        VParamItem* pNewItem = static_cast<VParamItem*>(pNodeParam->clone());
        lst[0]->appendRow(pNewItem);
    }
}

void PanelParamModel::onNodeParamsAboutToBeRemoved(const QModelIndex &parent, int first, int last)
{
    QStandardItemModel* pModel = qobject_cast<QStandardItemModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex& idxNodeParam = pModel->index(first, 0, parent);
    if (!idxNodeParam.isValid())
        return;

    VParamItem* pNodeParam = static_cast<VParamItem*>(pModel->itemFromIndex(idxNodeParam));
    VParamItem* parentItem = static_cast<VParamItem*>(pNodeParam->parent());
    const QString &parentName = parentItem->m_name;
    QList<QStandardItem *> lst;
    if (parentName == iotags::params::node_inputs)
    {
        lst = findItems(iotags::params::panel_inputs, Qt::MatchRecursive | Qt::MatchExactly);
    }
    else if (parentName == iotags::params::node_params)
    {
        lst = findItems(iotags::params::panel_params, Qt::MatchRecursive | Qt::MatchExactly);
    }
    else if (parentName == iotags::params::node_outputs)
    {
        lst = findItems(iotags::params::panel_outputs, Qt::MatchRecursive | Qt::MatchExactly);
    }
    ZASSERT_EXIT(lst.size() == 1);
    for (int row = 0; row < lst[0]->rowCount(); row++) {
        VParamItem *pChild = static_cast<VParamItem *>(lst[0]->child(row));
        if (pChild && pChild->m_name == pNodeParam->m_name) {
            lst[0]->removeRow(row);
        }
    }
}
