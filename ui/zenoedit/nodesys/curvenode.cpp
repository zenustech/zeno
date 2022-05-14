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

    //zeno::log_critical("MakeCurveNode add editbtn {}", pEditBtn);

    return pHLayout;
}

void MakeCurveNode::onEditClicked()
{
    PARAMS_INFO params = index().data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    PARAMS_INFO params2 = index().data(ROLE_PARAMETERS_NOT_DESC).value<PARAMS_INFO>();

    QString parstr = params2["UI_MakeCurve"].value.toString();

    QVector<CURVE_DATA> curves;

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

            curves.push_back({});
            auto &curve = curves.back();
            curve.key = key;
            curve.cycleType = cyctype;

            curve.rg.xFrom = L[i++].toFloat(&bOK);
            ZASSERT_EXIT(bOK);
            curve.rg.xTo = L[i++].toFloat(&bOK);
            ZASSERT_EXIT(bOK);
            curve.rg.yFrom = L[i++].toFloat(&bOK);
            ZASSERT_EXIT(bOK);
            curve.rg.yTo = L[i++].toFloat(&bOK);
            ZASSERT_EXIT(bOK);

            for (int j = 0; j < count; j++) {

                QPointF pt;
                pt.setX(L[i++].toFloat(&bOK));
                ZASSERT_EXIT(bOK);
                pt.setY(L[i++].toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                int cptype = L[i++].toInt(&bOK);
                ZASSERT_EXIT(bOK);

                QPointF pt1;
                pt1.setX(L[i++].toFloat(&bOK));
                ZASSERT_EXIT(bOK);
                pt1.setY(L[i++].toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                QPointF pt2;
                pt1.setX(L[i++].toFloat(&bOK));
                ZASSERT_EXIT(bOK);
                pt1.setY(L[i++].toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                curve.points.push_back({
                    pt, pt1, pt2, cptype,
                });
            }
        }
    }
    else
    {
        CURVE_RANGE rg;
        rg.xFrom = 0;
        rg.xTo = 1;
        rg.yFrom = 0;
        rg.yTo = 1;
        {
            curves.push_back({});
            auto &curve = curves.back();
            curve.key = "x";
            curve.cycleType = 0;
            curve.points.append({QPointF(rg.xFrom, rg.yTo), QPointF(0, 0), QPointF(0, 0), 0});
            curve.points.append({QPointF(rg.xTo, rg.yFrom), QPointF(0, 0), QPointF(0, 0), 0});
        }
        {
            curves.push_back({});
            auto &curve = curves.back();
            curve.key = "y";
            curve.cycleType = 0;
            curve.points.append({QPointF(rg.xFrom, rg.yFrom), QPointF(0, 0), QPointF(0, 0), 0});
            curve.points.append({QPointF(rg.xTo, rg.yTo), QPointF(0, 0), QPointF(0, 0), 0});
        }
        {
            curves.push_back({});
            auto &curve = curves.back();
            curve.key = "z";
            curve.cycleType = 0;
            curve.points.append({QPointF(rg.xFrom, rg.yTo / 2), QPointF(0, 0), QPointF(0, 0), 0});
            curve.points.append({QPointF(rg.xTo, rg.yTo / 2), QPointF(0, 0), QPointF(0, 0), 0});
        }
    }

    ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);

    for (auto const &curve: curves) {
        CurveModel *pModel = new CurveModel(curve.key, curve.rg, this);
        pModel->initItems(curve);
        pEditor->addCurve(pModel);
    }

    pEditor->show();
}
