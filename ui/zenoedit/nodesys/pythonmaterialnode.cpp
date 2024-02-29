#include "pythonmaterialnode.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/igraphsmodel.h>
#include "dialog/zmaterialinfosettingdlg.h"

PythonMaterialNode::PythonMaterialNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
}

PythonMaterialNode::~PythonMaterialNode()
{
}

ZGraphicsLayout* PythonMaterialNode::initCustomParamWidgets()
{
    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(false);
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
    pHLayout->addSpacing(100);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Execute", -1, QSizePolicy::Expanding);
    pEditBtn->setMinimumHeight(32);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onExecuteClicked()));

    pHLayout->addSpacing(10);
    ZenoParamPushButton* pGenerateBtn = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
    pGenerateBtn->setMinimumHeight(32);
    pHLayout->addItem(pGenerateBtn);
    connect(pGenerateBtn, SIGNAL(clicked()), this, SLOT(onGenerateClicked()));

    pVLayout->addLayout(pHLayout);

    pVLayout->addSpacing(10);

    ZGraphicsLayout* pHLayout1 = new ZGraphicsLayout(true);
    ZSimpleTextItem* pNameItem1 = new ZSimpleTextItem("sync");
    pNameItem1->setBrush(m_renderParams.socketClr.color());
    pNameItem1->setFont(m_renderParams.socketFont);
    pNameItem1->updateBoundingRect();

    pHLayout1->addItem(pNameItem1);

    pHLayout1->addSpacing(48);

    ZenoParamPushButton* pEditBtn1 = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pEditBtn1->setMinimumHeight(32);
    pHLayout1->addItem(pEditBtn1);
    connect(pEditBtn1, SIGNAL(clicked()), this, SLOT(onEditClicked()));
    pVLayout->addLayout(pHLayout1);
    return pVLayout;
}

void PythonMaterialNode::onExecuteClicked()
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = subgIndex();
    QModelIndex scriptIdx = pModel->paramIndex(subgIdx, index(), "script", true);
    ZASSERT_EXIT(scriptIdx.isValid());
    QString script = scriptIdx.data(ROLE_PARAM_VALUE).toString();
    MaterialMatchInfo info = getMatchInfo();
    QPointF pos = nodePos();
    script = script.arg(info.m_names, info.m_materialPath, info.m_keyWords, info.m_matchInputs, QString::number(pos.x()), QString::number(pos.y()));

    AppHelper::pythonExcute(script);
}


void PythonMaterialNode::onEditClicked()
{
    MaterialMatchInfo info = getMatchInfo();
    ZMaterialInfoSettingDlg::getMatchInfo(info);
    setMatchInfo(info);
}

void PythonMaterialNode::onGenerateClicked()
{
    AppHelper::generatePython(this->nodeId());
}

MaterialMatchInfo PythonMaterialNode::getMatchInfo()
{
    MaterialMatchInfo info;
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = subgIndex();
    QModelIndex nameIdx = pModel->paramIndex(subgIdx, index(), "nameList", true);
    ZASSERT_EXIT(nameIdx.isValid(), info);
    info.m_names = nameIdx.data(ROLE_PARAM_VALUE).toString();

    QModelIndex keyIdx = pModel->paramIndex(subgIdx, index(), "keyWords", true);
    ZASSERT_EXIT(keyIdx.isValid(), info);
    info.m_keyWords = keyIdx.data(ROLE_PARAM_VALUE).toString();

    QModelIndex pathIdx = pModel->paramIndex(subgIdx, index(), "materialPath", true);
    ZASSERT_EXIT(pathIdx.isValid(), info);
    info.m_materialPath= pathIdx.data(ROLE_PARAM_VALUE).toString();

    QModelIndex matchIdx = pModel->paramIndex(subgIdx, index(), "matchInputs", true);
    ZASSERT_EXIT(matchIdx.isValid(), info);
    info.m_matchInputs = matchIdx.data(ROLE_PARAM_VALUE).toString();
    return info;
}

void PythonMaterialNode::setMatchInfo(const MaterialMatchInfo& matInfo)
{
    PARAM_UPDATE_INFO info;
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    QModelIndex subgIdx = subgIndex();
    INPUT_SOCKETS inputs = index().data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    ZASSERT_EXIT(inputs.find("nameList") != inputs.end() && inputs.find("keyWords") != inputs.end()
        && inputs.find("matchInputs") != inputs.end() && inputs.find("materialPath") != inputs.end());
    const QString& nodeid = this->nodeId();
    //name list
    info.name = "nameList";
    info.oldValue = inputs[info.name].info.defaultValue;
    info.newValue = matInfo.m_names;
    pModel->updateSocketDefl(nodeid, info, subgIdx, true);
    //key words
    info.name = "keyWords";
    info.oldValue = inputs[info.name].info.defaultValue;
    info.newValue = matInfo.m_keyWords;
    pModel->updateSocketDefl(nodeid, info, subgIdx, true);
    //path
    info.name = "materialPath";
    info.oldValue = inputs[info.name].info.defaultValue;
    info.newValue = matInfo.m_materialPath;
    pModel->updateSocketDefl(nodeid, info, subgIdx, true);
    //match inputs info
    info.name = "matchInputs";
    info.oldValue = inputs[info.name].info.defaultValue;
    info.newValue = matInfo.m_matchInputs;
    pModel->updateSocketDefl(nodeid, info, subgIdx, true);
}