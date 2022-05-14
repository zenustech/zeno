#include "curvenode.h"
#include "../curvemap/zcurvemapeditor.h"
#include "../model/curvemodel.h"
#include <zenoui/util/cihou.h>
#include "util/log.h"


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

QGraphicsLinearLayout* MakeCurveNode::initCustomParamWidgets()
{
    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("curve", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    return pHLayout;
}

void MakeCurveNode::onEditClicked()
{
    PARAMS_INFO params = index().data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    PARAMS_INFO params2 = index().data(ROLE_PARAMETERS_NOT_DESC).value<PARAMS_INFO>();

    CURVE_RANGE rg;

    QString parstr = params2["UI_MakeCurve"].value.toString();

    QVector<QPointF> points, handlers;

    QStringList L = parstr.split(" ", QtSkipEmptyParts);
    if (!L.isEmpty())
    {
        bool bOK = false;
        int keycount = L[0].toInt(&bOK);
        ZASSERT_EXIT(bOK);

        int i = 1;
        for (int k = 0; k < keycount; k++)
        {
            QString key = L[i++];
            int cyctype = L[i++].toInt(&bOK);
            ZASSERT_EXIT(bOK);
            int count = L[i++].toInt(&bOK);
            ZASSERT_EXIT(bOK);
            float input_min = L[i++].toFloat(&bOK);
            ZASSERT_EXIT(bOK);
            float input_max = L[i++].toFloat(&bOK);
            ZASSERT_EXIT(bOK);
            float output_min = L[i++].toFloat(&bOK);
            ZASSERT_EXIT(bOK);
            float output_max = L[i++].toFloat(&bOK);
            ZASSERT_EXIT(bOK);

            QPointF pt;
            pt.setX(L[i++].toFloat(&bOK));
            ZASSERT_EXIT(bOK);
            pt.setY(L[i++].toFloat(&bOK));
            ZASSERT_EXIT(bOK);
            points.append(pt);

            int cptype = L[i++].toInt(&bOK);
            ZASSERT_EXIT(bOK);

            QPointF pt1;
            pt1.setX(L[i++].toFloat(&bOK));
            ZASSERT_EXIT(bOK);
            pt1.setY(L[i++].toFloat(&bOK));
            ZASSERT_EXIT(bOK);
            handlers.append(pt1);

            QPointF pt2;
            pt1.setX(L[i++].toFloat(&bOK));
            ZASSERT_EXIT(bOK);
            pt1.setY(L[i++].toFloat(&bOK));
            ZASSERT_EXIT(bOK);
            handlers.append(pt1);
        }
    }
    else
    {
        points.append(QPointF(rg.xFrom, rg.yFrom));
        points.append(QPointF(rg.xTo, rg.yTo));
        handlers.append(QPointF(0, 0));
        handlers.append(QPointF(0, 0));
        handlers.append(QPointF(0, 0));
        handlers.append(QPointF(0, 0));
    }

    ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);

    CurveModel *pModel = new CurveModel("x", rg, this);
    pModel->initItems(rg, points, handlers);
    pEditor->addCurve(pModel);

    CurveModel *pModel2 = new CurveModel("y", rg, this);
    pModel2->initItems(rg, {{rg.xFrom, rg.yTo}, {rg.xTo, rg.yFrom}}, {{0,0}, {0,0}, {0,0}, {0,0}});
    pEditor->addCurve(pModel2);

    CurveModel *pModel3 = new CurveModel("z", rg, this);
    pModel3->initItems(rg, {{rg.xFrom, rg.yTo / 2}, {rg.xTo, rg.yTo / 2}}, {{0,0}, {0,0}, {0,0}, {0,0}});
    pEditor->addCurve(pModel3);

    pEditor->show();
}
