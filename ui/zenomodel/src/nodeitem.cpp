#include "nodeitem.h"
#include "uihelper.h"
#include "variantptr.h"
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/panelparammodel.h>
#include "subgraphmodel.h"


NodeItem::NodeItem(QObject* parent)
    : QObject(parent)
    , options(0)
    , bCollasped(false)
    , type(NORMAL_NODE)
    , panelParams(nullptr)
    , nodeParams(nullptr)
    , treeItem(nullptr)
{
}

QModelIndex NodeItem::nodeIdx() const
{
    if (SubGraphModel* pSubg = qobject_cast<SubGraphModel *>(parent()))
    {
        return pSubg->index(this->objid);
    }
    else if (treeItem)
    {
        return treeItem->index();
    }
}


///////////////////////////////////////////////////////////////////////////
TreeNodeItem::TreeNodeItem(const NODE_DATA& nodeData, IGraphsModel* pGraphsModel)
{
    m_item = new NodeItem;
    m_item->treeItem = this;
    m_item->objid = nodeData[ROLE_OBJID].toString();
    m_item->objCls = nodeData[ROLE_OBJNAME].toString();
    m_item->customName = nodeData[ROLE_CUSTOM_OBJNAME].toString();
    m_item->viewpos = nodeData[ROLE_OBJPOS].toPointF();
    m_item->bCollasped = nodeData[ROLE_COLLASPED].toBool();
    m_item->options = nodeData[ROLE_OPTIONS].toInt();
    m_item->type = (NODE_TYPE)nodeData[ROLE_NODETYPE].toInt();
    m_item->paramNotDesc = nodeData[ROLE_PARAMS_NO_DESC].value<PARAMS_INFO>();
    m_item->nodeParams = new NodeParamModel(pGraphsModel, false, m_item);

    INPUT_SOCKETS inputs = nodeData[ROLE_INPUTS].value<INPUT_SOCKETS>();
    PARAMS_INFO params = nodeData[ROLE_PARAMETERS].value<PARAMS_INFO>();
    OUTPUT_SOCKETS outputs = nodeData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();

    m_item->nodeParams->setInputSockets(inputs);
    m_item->nodeParams->setParams(params);
    m_item->nodeParams->setOutputSockets(outputs);

    VPARAM_INFO panelInfo;
    if (nodeData.find(ROLE_CUSTOMUI_PANEL_IO) != nodeData.end()) {
        panelInfo = nodeData[ROLE_CUSTOMUI_PANEL_IO].value<VPARAM_INFO>();
    }
    m_item->panelParams = new PanelParamModel(m_item->nodeParams, panelInfo, pGraphsModel, m_item);
}

TreeNodeItem::~TreeNodeItem()
{
    delete m_item;
    m_item = nullptr;
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
        case Qt::DisplayRole:       return m_item->objid;
        case ROLE_OBJID:            return m_item->objid;
        case ROLE_OBJNAME:          return m_item->objCls;
        case ROLE_CUSTOM_OBJNAME:   return m_item->customName;
        case ROLE_OBJDATA:          return QVariant::fromValue(TreeNodeItem::item2NodeData(m_item));
        case ROLE_NODETYPE:         return m_item->type;
        case ROLE_INPUTS:
        {
            //legacy interface.
            if (!m_item->nodeParams)
                return QVariant();

            INPUT_SOCKETS inputs;
            m_item->nodeParams->getInputSockets(inputs);
            return QVariant::fromValue(inputs);
        }
        case ROLE_OUTPUTS:
        {
            if (!m_item->nodeParams)
                return QVariant();

            OUTPUT_SOCKETS outputs;
            m_item->nodeParams->getOutputSockets(outputs);
            return QVariant::fromValue(outputs);
        }
        case ROLE_PARAMETERS:
        {
            if (!m_item->nodeParams)
                return QVariant();

            PARAMS_INFO params;
            m_item->nodeParams->getParams(params);
            return QVariant::fromValue(params);
        }
        case ROLE_COLLASPED:
        {
            return m_item->bCollasped;
        }
        case ROLE_OPTIONS:
        {
            return m_item->options;
        }
        case ROLE_OBJPOS:
        {
            return m_item->viewpos;
        }
        case ROLE_PANEL_PARAMS:
        {
            return QVariantPtr<ViewParamModel>::asVariant(m_item->panelParams);
        }
        case ROLE_NODE_PARAMS:
        {
            return QVariantPtr<QStandardItemModel>::asVariant(m_item->nodeParams);
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
            const QString& path = parentItem->data(ROLE_OBJPATH).toString() + "/" + m_item->objid;
            return path;
        }
        case ROLE_CUSTOMUI_PANEL_IO: {
            VPARAM_INFO root = m_item->panelParams->exportParams();
            return QVariant::fromValue(root);
        }
        case ROLE_PARAMS_NO_DESC: {
            return QVariant::fromValue(m_item->paramNotDesc);
        }
        case ROLE_SUBGRAPH_IDX: {
            QStandardItem* parentItem = this->parent();
            if (parentItem) {
                return parentItem->index();
            }
            return QVariant();
        }
        case ROLE_NODE_IDX: {
            return index();
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

void TreeNodeItem::addNode(const NODE_DATA& data, IGraphsModel* pModel)
{
    TreeNodeItem* pNewItem = new TreeNodeItem(data, pModel);
    appendRow(pNewItem);
}

void TreeNodeItem::appendRow(TreeNodeItem* pChildItem)
{
    ZASSERT_EXIT(pChildItem);
    const QString& ident = pChildItem->objName();
    m_ident2row[ident] = rowCount();
    QStandardItem::appendRow(pChildItem);
}

void TreeNodeItem::removeNode(const QString& ident, IGraphsModel* pModel)
{
    int row = id2Row(ident);
    if (row == -1)
        return;

    const QModelIndex& subgIdx = this->index();
    ZASSERT_EXIT(subgIdx.isValid());
    TreeNodeItem* pNode = this->childItem(ident);
    ZASSERT_EXIT(pNode);

    if (pNode->m_item->panelParams)
    {
        pNode->m_item->panelParams->clear();
        delete pNode->m_item->panelParams;
        pNode->m_item->panelParams = nullptr;
    }
    if (pNode->m_item->nodeParams)
    {
        pNode->m_item->nodeParams->clearParams();
        delete pNode->m_item->nodeParams;
        pNode->m_item->nodeParams = nullptr;
    }

    for (int r = row + 1; r < rowCount(); r++)
    {
        TreeNodeItem* pChildItem = static_cast<TreeNodeItem*>(child(r));
        const QString& id = pChildItem->data(ROLE_OBJID).toString();
        ZASSERT_EXIT(m_ident2row.find(id) != m_ident2row.end());
        m_ident2row[id] = r - 1;
    }

    ZASSERT_EXIT(m_ident2row.find(ident) != m_ident2row.end());
    m_ident2row.remove(ident);
    this->removeRow(row);

    pModel->markDirty();
}

NODE_DATA TreeNodeItem::expData() const
{
    NODE_DATA data;
    data[ROLE_OBJID] = m_item->objid;
    data[ROLE_OBJNAME] = m_item->objCls;
    data[ROLE_CUSTOM_OBJNAME] = m_item->customName;
    data[ROLE_OBJPOS] = m_item->viewpos;
    data[ROLE_COLLASPED] = m_item->bCollasped;
    data[ROLE_OPTIONS] = m_item->options;
    data[ROLE_NODETYPE] = m_item->type;

    INPUT_SOCKETS inputs;
    OUTPUT_SOCKETS outputs;
    PARAMS_INFO params;

    m_item->nodeParams->getInputSockets(inputs);
    m_item->nodeParams->getParams(params);
    m_item->nodeParams->getOutputSockets(outputs);

    data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
    data[ROLE_PARAMETERS] = QVariant::fromValue(params);
    data[ROLE_PANEL_PARAMS] = QVariantPtr<ViewParamModel>::asVariant(m_item->panelParams);
    data[ROLE_PARAMS_NO_DESC] = QVariant::fromValue(m_item->paramNotDesc);

    return data;
}

void TreeNodeItem::setData(const QVariant& value, int role)
{
    switch (role)
    {
        case ROLE_OBJNAME:
        {
            m_item->objCls = value.toString();
            break;
        }
        case ROLE_CUSTOM_OBJNAME:
        {
            TreeNodeItem* parentItem = static_cast<TreeNodeItem*>(this->parent());
            ZASSERT_EXIT(parentItem != nullptr);
            bool isValid = parentItem->checkCustomName(value.toString());
            if (isValid)
                m_item->customName = value.toString();
            else
                return;
            break;
        }
        case ROLE_INPUTS:
        {
            INPUT_SOCKETS inputs = value.value<INPUT_SOCKETS>();
            if (inputs.empty())
                return;

            ZASSERT_EXIT(m_item->nodeParams);
            for (QString name : inputs.keys()) {
                const INPUT_SOCKET &inSocket = inputs[name];
                m_item->nodeParams->setAddParam(PARAM_INPUT, name, inSocket.info.type, inSocket.info.defaultValue,
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

            ZASSERT_EXIT(m_item->nodeParams);
            for (QString name : outputs.keys()) {
                const OUTPUT_SOCKET &outSocket = outputs[name];
                m_item->nodeParams->setAddParam(PARAM_OUTPUT, name, outSocket.info.type, outSocket.info.defaultValue,
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

            ZASSERT_EXIT(m_item->nodeParams);
            for (QString name : params.keys()) {
                const PARAM_INFO &param = params[name];
                m_item->nodeParams->setAddParam(PARAM_PARAM, name, param.typeDesc, param.value, param.control,
                                             param.controlProps, SOCKPROP_UNKNOWN, DICTPANEL_INFO(), param.toolTip);
            }
            break;
        }
        case ROLE_CUSTOMUI_PANEL_IO:
        {
            const VPARAM_INFO &invisibleRoot = value.value<VPARAM_INFO>();
            ZASSERT_EXIT(m_item->panelParams);
            m_item->panelParams->importPanelParam(invisibleRoot);
            break;
        }
        case ROLE_COLLASPED:
        {
            m_item->bCollasped = value.toBool();
            break;
        }
        case ROLE_OPTIONS: {
            m_item->options = value.toInt();
            break;
        }
        case ROLE_OBJPOS: {
            m_item->viewpos = value.toPointF();
            break;
        }
        case ROLE_PARAMS_NO_DESC: {
            m_item->paramNotDesc = value.value<PARAMS_INFO>();
            break;
        }
    }
}

QString TreeNodeItem::objClass() const
{
    return data(ROLE_OBJNAME).toString();
}

QString TreeNodeItem::objName() const
{
    return data(ROLE_OBJID).toString();
}

QModelIndex TreeNodeItem::childIndex(const QString& ident) const
{
    int row = id2Row(ident);
    if (row < 0 || row >= rowCount())
        return QModelIndex();
    return child(row)->index();
}

TreeNodeItem* TreeNodeItem::childItem(const QString& ident)
{
    int row = id2Row(ident);
    if (row < 0 || row >= rowCount())
        return nullptr;
    return static_cast<TreeNodeItem*>(child(row));
}
