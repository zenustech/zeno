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
    PARAMS_INFO &params = index().data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    PARAMS_INFO &params2 = index().data(ROLE_PARAMETERS_NOT_DESC).value<PARAMS_INFO>();

    QVariant &parval = params2["UI_MakeCurve"].value;
    QString parstr = parval.toString();

    QVector<CURVE_DATA> curves;

    QStringList L = parstr.split(" ", QtSkipEmptyParts);
    if (!L.isEmpty())
    {
        bool bOK = false;
        int keycount = L[0].toInt(&bOK);
        ZASSERT_EXIT(bOK);

        auto Lp = L.constBegin();
        for (int k = 0; k < keycount; k++)
        {
            ZASSERT_EXIT(Lp != L.constEnd());
            QString key = *Lp++;

            ZASSERT_EXIT(Lp != L.constEnd());
            int cyctype = Lp++->toInt(&bOK);
            ZASSERT_EXIT(bOK);

            ZASSERT_EXIT(Lp != L.constEnd());
            int count = Lp++->toInt(&bOK);
            ZASSERT_EXIT(bOK);

            curves.push_back({});
            auto &curve = curves.back();
            curve.key = key;
            curve.cycleType = cyctype;

            ZASSERT_EXIT(Lp != L.constEnd());
            curve.rg.xFrom = Lp++->toFloat(&bOK);
            ZASSERT_EXIT(bOK);

            ZASSERT_EXIT(Lp != L.constEnd());
            curve.rg.xTo = Lp++->toFloat(&bOK);
            ZASSERT_EXIT(bOK);

            ZASSERT_EXIT(Lp != L.constEnd());
            curve.rg.yFrom = Lp++->toFloat(&bOK);
            ZASSERT_EXIT(bOK);

            ZASSERT_EXIT(Lp != L.constEnd());
            curve.rg.yTo = Lp++->toFloat(&bOK);
            ZASSERT_EXIT(bOK);

            for (int j = 0; j < count; j++) {

                QPointF pt;

                ZASSERT_EXIT(Lp != L.constEnd());
                pt.setX(Lp++->toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                ZASSERT_EXIT(Lp != L.constEnd());
                pt.setY(Lp++->toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                ZASSERT_EXIT(Lp != L.constEnd());
                int cptype = Lp++->toInt(&bOK);
                ZASSERT_EXIT(bOK);

                QPointF pt1;

                ZASSERT_EXIT(Lp != L.constEnd());
                pt1.setX(Lp++->toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                ZASSERT_EXIT(Lp != L.constEnd());
                pt1.setY(Lp++->toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                QPointF pt2;

                ZASSERT_EXIT(Lp != L.constEnd());
                pt1.setX(Lp++->toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                ZASSERT_EXIT(Lp != L.constEnd());
                pt1.setY(Lp++->toFloat(&bOK));
                ZASSERT_EXIT(bOK);

                curve.points.push_back({
                    pt, pt1, pt2, cptype,
                });
            }
        }
    }
    else
    {
        {
            curves.push_back({});
            auto &curve = curves.back();
            curve.key = "x";
            curve.cycleType = 0;
            CURVE_RANGE rg;
            rg.xFrom = 0;
            rg.yFrom = 0;
            rg.xTo = 1;
            rg.yTo = 1;
            curve.rg = rg;
            curve.points.append({QPointF(rg.xFrom, rg.yFrom), QPointF(0, 0), QPointF(0, 0), 0});
            curve.points.append({QPointF(rg.xTo, rg.yTo), QPointF(0, 0), QPointF(0, 0), 0});
        }
    }

    ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);

    for (auto const &curve: curves) {
        CurveModel *pModel = new CurveModel(curve.key, curve.rg, this);
        pModel->initItems(curve);
        pEditor->addCurve(pModel);
    }

    pEditor->show();
    connect(pEditor, &ZCurveMapEditor::finished, this, [&parval, pEditor] (int result) {
        ZENO_P(result);
        QString parstr = "";

#if 0  // TODO: LUZH please complete the following pEditor->getXXX()
        int keycount = pEditor->getCurvesCount();
        parstr.append(QString::number(keycount));
        for (int k = 0; k < keycount; k++) {
            QString key = pEditor->getCurveKey(k);
            parstr.append(' ');
            parstr.append(key);
            int cyctype = 0;
            parstr.append(' ');
            parstr.append(QString::number(cyctype));
            int count = pEditor->getCurvePointCount(k);
            parstr.append(' ');
            parstr.append(QString::number(count));
            CURVE_RANGE rg = pEditor->getCurveRange(k);
            parstr.append(' ');
            parstr.append(QString::number(rg.xFrom));
            parstr.append(' ');
            parstr.append(QString::number(rg.xTo));
            parstr.append(' ');
            parstr.append(QString::number(rg.yFrom));
            parstr.append(' ');
            parstr.append(QString::number(rg.yTo));
            for (int i = 0; i < count; i++) {
                QVector2D pt = pEditor->getCurvePointPosition(k, i);
                parstr.append(' ');
                parstr.append(QString::number(pt.x()));
                parstr.append(' ');
                parstr.append(QString::number(pt.y()));
                QVector2D lh = pEditor->getCurvePointLeftHandler(k, i);
                parstr.append(' ');
                parstr.append(QString::number(lh.x()));
                parstr.append(' ');
                parstr.append(QString::number(lh.y()));
                QVector2D rh = pEditor->getCurvePointRightHandler(k, i);
                parstr.append(' ');
                parstr.append(QString::number(rh.x()));
                parstr.append(' ');
                parstr.append(QString::number(rh.y()));
            }
        }
        zeno::log_debug("{}", parstr.toStdString());
#endif

        parval.setValue(parstr);
    });
}
