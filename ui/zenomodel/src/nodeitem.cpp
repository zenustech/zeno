#include "nodeitem.h"
#include "uihelper.h"
#include "variantptr.h"
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/panelparammodel.h>


TreeNodeItem::TreeNodeItem(const TreeNodeItem&)
{

}

TreeNodeItem::~TreeNodeItem()
{

}

NODE_DATA TreeNodeItem::item2NodeData(const NodeItem& item)
{
    NODE_DATA data;
    data[ROLE_OBJID] = item.objid;
    data[ROLE_OBJNAME] = item.objCls;
    data[ROLE_CUSTOM_OBJNAME] = item.customName;
    data[ROLE_OBJPOS] = item.viewpos;
    data[ROLE_COLLASPED] = item.bCollasped;
    data[ROLE_OPTIONS] = item.options;
    data[ROLE_NODETYPE] = item.type;

    INPUT_SOCKETS inputs;
    OUTPUT_SOCKETS outputs;
    PARAMS_INFO params;

    item.nodeParams->getInputSockets(inputs);
    item.nodeParams->getParams(params);
    item.nodeParams->getOutputSockets(outputs);

    data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
    data[ROLE_PARAMETERS] = QVariant::fromValue(params);
    data[ROLE_PANEL_PARAMS] = QVariantPtr<ViewParamModel>::asVariant(item.panelParams);
    data[ROLE_PARAMS_NO_DESC] = QVariant::fromValue(item.paramNotDesc);

    return data;
}


QVariant TreeNodeItem::data(int role) const
{
    switch (role)
    {
        case ROLE_OBJID:            return m_item.objid;
        case ROLE_OBJNAME:          return m_item.objCls;
        case ROLE_CUSTOM_OBJNAME:   return m_item.customName;
        case ROLE_OBJDATA:          return QVariant::fromValue(TreeNodeItem::item2NodeData(m_item));
        case ROLE_NODETYPE:         return m_item.type;
        case ROLE_INPUTS:
        {
            //legacy interface.
            if (!m_item.nodeParams)
                return QVariant();

            INPUT_SOCKETS inputs;
            m_item.nodeParams->getInputSockets(inputs);
            return QVariant::fromValue(inputs);
        }
        case ROLE_OUTPUTS:
        {
            if (!m_item.nodeParams)
                return QVariant();

            OUTPUT_SOCKETS outputs;
            m_item.nodeParams->getOutputSockets(outputs);
            return QVariant::fromValue(outputs);
        }
        case ROLE_PARAMETERS:
        {
            if (!m_item.nodeParams)
                return QVariant();

            PARAMS_INFO params;
            m_item.nodeParams->getParams(params);
            return QVariant::fromValue(params);
        }
        case ROLE_COLLASPED:
        {
            return m_item.bCollasped;
        }
        case ROLE_OPTIONS:
        {
            return m_item.options;
        }
        case ROLE_OBJPOS:
        {
            return m_item.viewpos;
        }
        case ROLE_PANEL_PARAMS:
        {
            return QVariantPtr<ViewParamModel>::asVariant(m_item.panelParams);
        }
        case ROLE_NODE_PARAMS:
        {
            return QVariantPtr<QStandardItemModel>::asVariant(m_item.nodeParams);
        }
        case ROLE_OBJPATH:
        {
            //format like: /main/[subgraph-A]/[subg-B]/objid-xxx
            //1. /main is a fix part, because all nodes starts from main subgraph.
            //2. any objpath does not ends with `/`
            QStandardItem* parentItem = this->parent();
            if (parentItem == nullptr) {
                return "/main";
            }
            const QString& path = parentItem->data(ROLE_OBJPATH).toString() + "/" + m_item.objid;
            return path;
        }
        case ROLE_CUSTOMUI_PANEL_IO: {
            VPARAM_INFO root = m_item.panelParams->exportParams();
            return QVariant::fromValue(root);
        }
        case ROLE_PARAMS_NO_DESC: {
            return QVariant::fromValue(m_item.paramNotDesc);
        }
        case ROLE_SUBGRAPH_IDX: {
            QStandardItem* parentItem = this->parent();
            if (parentItem) {
                return parentItem->index();
            }
            return QVariant();
        }
        default:
            return QVariant();
    }
}

bool TreeNodeItem::checkCustomName(const QString& name)
{
    //only check current layer.
    for (int r = 0; r < rowCount(); r++)
    {
        const QString& customName = this->child(r)->data(ROLE_CUSTOM_OBJNAME).toString();
        if (name == customName)
            return false;
    }
    return true;
}

void TreeNodeItem::setData(const QVariant& value, int role)
{
    switch (role)
    {
        case ROLE_OBJNAME:
        {
            m_item.objCls = value.toString();
            break;
        }
        case ROLE_CUSTOM_OBJNAME:
        {
            TreeNodeItem* parentItem = static_cast<TreeNodeItem*>(this->parent());
            ZASSERT_EXIT(parentItem != nullptr);
            bool isValid = parentItem->checkCustomName(value.toString());
            if (isValid)
                m_item.customName = value.toString();
            else
                return;
            break;
        }
        case ROLE_INPUTS:
        {
            INPUT_SOCKETS inputs = value.value<INPUT_SOCKETS>();
            if (inputs.empty())
                return;

            ZASSERT_EXIT(m_item.nodeParams);
            for (QString name : inputs.keys()) {
                const INPUT_SOCKET &inSocket = inputs[name];
                m_item.nodeParams->setAddParam(PARAM_INPUT, name, inSocket.info.type, inSocket.info.defaultValue,
                                             inSocket.info.control, inSocket.info.ctrlProps,
                                             (SOCKET_PROPERTY)inSocket.info.sockProp, inSocket.info.dictpanel,
                                             inSocket.info.toolTip);
            }
            break;
        }
        case ROLE_OUTPUTS:
        {
            OUTPUT_SOCKETS outputs = value.value<OUTPUT_SOCKETS>();
            if (outputs.empty())
                return;

            ZASSERT_EXIT(m_item.nodeParams);
            for (QString name : outputs.keys()) {
                const OUTPUT_SOCKET &outSocket = outputs[name];
                m_item.nodeParams->setAddParam(PARAM_OUTPUT, name, outSocket.info.type, outSocket.info.defaultValue,
                                             outSocket.info.control, outSocket.info.ctrlProps,
                                             (SOCKET_PROPERTY)outSocket.info.sockProp, outSocket.info.dictpanel,
                                             outSocket.info.toolTip);
            }
            break;
        }
        case ROLE_PARAMETERS:
        {
            PARAMS_INFO params = value.value<PARAMS_INFO>();
            if (params.empty())
                return;

            ZASSERT_EXIT(m_item.nodeParams);
            for (QString name : params.keys()) {
                const PARAM_INFO &param = params[name];
                m_item.nodeParams->setAddParam(PARAM_PARAM, name, param.typeDesc, param.value, param.control,
                                             param.controlProps, SOCKPROP_UNKNOWN, DICTPANEL_INFO(), param.toolTip);
            }
            break;
        }
        case ROLE_CUSTOMUI_PANEL_IO:
        {
            const VPARAM_INFO &invisibleRoot = value.value<VPARAM_INFO>();
            ZASSERT_EXIT(m_item.panelParams);
            m_item.panelParams->importPanelParam(invisibleRoot);
            break;
        }
        case ROLE_COLLASPED:
        {
            m_item.bCollasped = value.toBool();
            break;
        }
        case ROLE_OPTIONS: {
            m_item.options = value.toInt();
            break;
        }
        case ROLE_OBJPOS: {
            m_item.viewpos = value.toPointF();
            break;
        }
        case ROLE_PARAMS_NO_DESC: {
            m_item.paramNotDesc = value.value<PARAMS_INFO>();
            break;
        }
    }
}