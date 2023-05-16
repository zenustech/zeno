#include <QObject>
#include <QtWidgets>
#include <rapidjson/document.h>

#include "treeacceptor.h"
#include "graphstreemodel.h"
#include "modelrole.h"
#include <zeno/utils/logger.h>
#include "magic_enum.hpp"
#include "zassert.h"
#include "uihelper.h"
#include "variantptr.h"
#include "dictkeymodel.h"


TreeAcceptor::TreeAcceptor(GraphsTreeModel* pModel, bool bImport)
    : m_pModel(pModel)
    , m_currentGraph(nullptr)
    , m_bImport(bImport)
{
}

//IAcceptor
bool TreeAcceptor::setLegacyDescs(const rapidjson::Value& graphObj, const NODE_DESCS& nodesParams)
{
    return false;
}

void TreeAcceptor::BeginSubgraph(const QString& name)
{

}

bool TreeAcceptor::setCurrentSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx)
{
    return false;
}

void TreeAcceptor::EndSubgraph()
{

}

void TreeAcceptor::EndGraphs()
{

}

void TreeAcceptor::setFilePath(const QString& fileName)
{

}

void TreeAcceptor::switchSubGraph(const QString &graphName)
{

}

bool TreeAcceptor::addNode(const QString &nodeid, const QString &name, const QString &customName,
             const NODE_DESCS &descriptors)
{
    return false;
}

void TreeAcceptor::setViewRect(const QRectF &rc)
{

}

void TreeAcceptor::setSocketKeys(const QString &id, const QStringList &keys)
{

}

void TreeAcceptor::initSockets(const QString &id, const QString &name, const NODE_DESCS &descs)
{

}

void TreeAcceptor::addDictKey(const QString &id, const QString &keyName, bool bInput)
{

}

void TreeAcceptor::addSocket(bool bInput, const QString &ident, const QString &sockName,
                             const QString &sockProperty)
{

}

void TreeAcceptor::setInputSocket2(
                    const QString &nodeCls,
                    const QString &inNode,
                    const QString &inSock,
                    const QString &outLinkPath,
                    const QString &sockProperty,
                    const rapidjson::Value &defaultValue)
{

}

void TreeAcceptor::setInputSocket(
                    const QString &nodeCls,
                    const QString &inNode,
                    const QString &inSock,
                    const QString &outNode,
                    const QString &outSock,
                    const rapidjson::Value &defaultValue)
{
}

void TreeAcceptor::addInnerDictKey(
                    bool bInput,
                    const QString &inNode,
                    const QString &inSock,
                    const QString &keyName,
                    const QString &link)
{

}

void TreeAcceptor::setDictPanelProperty(
                    bool bInput,
                    const QString &ident,
                    const QString &sockName,
                    bool bCollasped)
{

}

void TreeAcceptor::setControlAndProperties(
                    const QString &nodeCls,
                    const QString &inNode,
                    const QString &inSock,
                    PARAM_CONTROL control,
                    const QVariant &ctrlProperties)
{

}

void TreeAcceptor::setToolTip(PARAM_CLASS cls, const QString &inNode, const QString &inSock,
                              const QString &toolTip)
{

}

void TreeAcceptor::setParamValue(const QString &id, const QString &nodeCls, const QString &name,
                   const rapidjson::Value &value)
{

}

void TreeAcceptor::setParamValue2(const QString &id, const QString &noCls, const PARAMS_INFO &params)
{

}

void TreeAcceptor::setPos(const QString &id, const QPointF &pos)
{

}

void TreeAcceptor::setOptions(const QString &id, const QStringList &options)
{

}

void TreeAcceptor::setColorRamps(const QString &id, const COLOR_RAMPS &colorRamps)
{

}

void TreeAcceptor::setBlackboard(const QString &id, const BLACKBOARD_INFO &blackboard)
{

}

void TreeAcceptor::setTimeInfo(const TIMELINE_INFO &info)
{

}

TIMELINE_INFO TreeAcceptor::timeInfo() const
{
    return TIMELINE_INFO();
}

void TreeAcceptor::setLegacyCurve(const QString &id, const QVector<QPointF> &pts,
                    const QVector<QPair<QPointF, QPointF>> &hdls)
{

}

QObject *TreeAcceptor::currGraphObj()
{
    return nullptr;
}

void TreeAcceptor::endInputs(const QString &id, const QString &nodeCls)
{

}

void TreeAcceptor::endParams(const QString &id, const QString &nodeCls)
{

}

void TreeAcceptor::addCustomUI(const QString &id, const VPARAM_INFO &invisibleRoot)
{

}

void TreeAcceptor::setIOVersion(zenoio::ZSG_VERSION versio)
{

}
