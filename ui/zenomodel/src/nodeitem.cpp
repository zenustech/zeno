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
    m_item->objid = nodeData.ident;
    m_item->objCls = nodeData.nodeCls;
    m_item->customName = nodeData.customName;
    m_item->viewpos = nodeData.pos;
    m_item->bCollasped = nodeData.bCollasped;
    m_item->options = nodeData.options;
    m_item->type = nodeData.type;
    m_item->paramNotDesc = nodeData.parmsNotDesc;
    m_item->nodeParams = new NodeParamModel(pGraphsModel, false, m_item);
    m_item->nodeParams->setInputSockets(nodeData.inputs);
    m_item->nodeParams->setParams(nodeData.params);
    m_item->nodeParams->setOutputSockets(nodeData.outputs);

    VPARAM_INFO panelInfo = nodeData.customPanel;
    m_item->panelParams = new PanelParamModel(m_item->nodeParams, panelInfo, pGraphsModel, m_item);

    for (QString ident : nodeData.children.keys())
    {
        const NODE_DATA& dat = nodeData.children[ident];
        appendRow(new TreeNodeItem(dat, pGraphsModel));
    }
}

TreeNodeItem::~TreeNodeItem()
{
    delete m_item;
    m_item = nullptr;
}

QVariant TreeNodeItem::data(int role) const
{
    switch (role)
    {
        case Qt::DisplayRole: 
        {
            if (m_item->objid != "main")
            {
                QString id = m_item->objid.left(m_item->objid.indexOf("-"));
                return m_item->objCls + QString(" (%1)").arg(id);
            }
            return m_item->objid;
        }
        case ROLE_OBJID:            return m_item->objid;
        case ROLE_OBJNAME:          return m_item->objCls;
        case ROLE_CUSTOM_OBJNAME:   return m_item->customName;
        case ROLE_OBJDATA:          return QVariant::fromValue(expData());
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
    data.ident = m_item->objid;
    data.nodeCls = m_item->objCls;
    data.customName = m_item->customName;
    data.pos = m_item->viewpos;
    data.bCollasped = m_item->bCollasped;
    data.options = m_item->options;
    data.type = m_item->type;

    m_item->nodeParams->getInputSockets(data.inputs);
    m_item->nodeParams->getParams(data.params);
    m_item->nodeParams->getOutputSockets(data.outputs);

    data.customPanel = m_item->panelParams->exportParams();
    data.parmsNotDesc = m_item->paramNotDesc;

    for (int r = 0; r < rowCount(); r++)
    {
        TreeNodeItem* pChild = static_cast<TreeNodeItem*>(child(r));
        const QString& ident = pChild->m_item->objid;
        data.children.insert(ident, pChild->expData());
    }
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
    QStandardItem::setData(value, role);
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
