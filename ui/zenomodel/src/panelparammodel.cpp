#include "panelparammodel.h"
#include "nodeparammodel.h"
#include "vparamitem.h"
#include "modelrole.h"


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
}

PanelParamModel::~PanelParamModel()
{
}

void PanelParamModel::initParams(NodeParamModel* nodeParams)
{
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
    VParamItem *pRoot = new VParamItem(VPARAM_ROOT, "root");
    pRoot->setEditable(false);

    VParamItem *pTab = new VParamItem(VPARAM_TAB, "Default");
    {
        VParamItem *pInputsGroup = new VParamItem(VPARAM_GROUP, "In Sockets");
        VParamItem *paramsGroup = new VParamItem(VPARAM_GROUP, "Parameters");
        VParamItem *pOutputsGroup = new VParamItem(VPARAM_GROUP, "Out Sockets");

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