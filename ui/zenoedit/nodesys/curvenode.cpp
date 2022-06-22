#include "curvenode.h"
#include "../curvemap/zcurvemapeditor.h"
#include <zenoui/model/curvemodel.h>
#include <zenoui/model/variantptr.h>
#include <zenoui/util/cihou.h>
#include "util/log.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"


MakeCurveNode::MakeCurveNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
}

MakeCurveNode::~MakeCurveNode()
{

}

QGraphicsLayout* MakeCurveNode::initParams()
{
    return ZenoNode::initParams();
}

void MakeCurveNode::initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& name, const PARAM_INFO& param)
{
    ZenoNode::initParam(ctrl, pParamLayout, name, param);
}

/*
QGraphicsLinearLayout* MakeCurveNode::initCustomParamWidgets()
{
    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("curve", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    //zeno::log_critical("MakeCurveNode add editbtn {}", pEditBtn);

    return pHLayout;
}

void MakeCurveNode::onEditClicked()
{
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();

    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);

    CurveModel *pModel = nullptr;
    PARAM_INFO& param = params["UI_MakeCurve"];
    pModel = QVariantPtr<CurveModel>::asPtr(param.value);

    if (!pModel)
    {
        pModel = curve_util::deflModel(pGraphsModel);
        param.value = QVariantPtr<CurveModel>::asVariant(pModel);

        PARAM_UPDATE_INFO paramInfo;
        paramInfo.name = "UI_MakeCurve";
        paramInfo.newValue = param.value;

        const QString& id = index().data(ROLE_OBJID).toString();
        pGraphsModel->updateParamNotDesc(id, paramInfo, subGraphIndex(), false);
    }
    
    ZASSERT_EXIT(pModel);

    ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);
    pEditor->setAttribute(Qt::WA_DeleteOnClose);
    pEditor->addCurve(pModel);
    pEditor->show();
}
*/
