#if 0
#include "curvemapnode.h"
#include "../curvemap/zcurvemapeditor.h"
#include <zenoui/model/curvemodel.h>
#include <zenoui/util/cihou.h>
#include "util/log.h"


MakeCurvemapNode::MakeCurvemapNode(const NodeUtilParam& params, QGraphicsItem* parent)
	: ZenoNode(params, parent)
{

}

MakeCurvemapNode::~MakeCurvemapNode()
{

}

QGraphicsLayout* MakeCurvemapNode::initParams()
{
	return ZenoNode::initParams();
}

void MakeCurvemapNode::initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& name, const PARAM_INFO& param)
{
	ZenoNode::initParam(ctrl, pParamLayout, name, param);
}

QGraphicsLinearLayout* MakeCurvemapNode::initCustomParamWidgets()
{
	QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

	ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("curve", m_renderParams.paramFont, m_renderParams.paramClr.color());
	pHLayout->addItem(pNameItem);

	ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
	pHLayout->addItem(pEditBtn);
	connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

	return pHLayout;
}

void MakeCurvemapNode::onEditClicked()
{
	PARAMS_INFO params = index().data(ROLE_PARAMETERS).value<PARAMS_INFO>();
	PARAMS_INFO params2 = index().data(ROLE_PARAMETERS_NOT_DESC).value<PARAMS_INFO>();
	
	if (params.contains("input_min") &&
		params.contains("input_max") &&
		params.contains("output_min") &&
		params.contains("output_max"))
	{
		CURVE_RANGE rg;

		rg.xFrom = params["input_min"].value.toFloat();
		rg.xTo = params["input_max"].value.toFloat();
		rg.yFrom = params["output_min"].value.toFloat();
		rg.yTo = params["output_max"].value.toFloat();

		QString pointsStr = params2["_POINTS"].value.toString();
		QString handlersStr = params2["_HANDLERS"].value.toString();

		QVector<QPointF> points, handlers;

		QStringList L = pointsStr.split(" ", QtSkipEmptyParts);
		if (!L.isEmpty())
		{
			int n = 0;
			bool bOK = false;
			n = L[0].toInt(&bOK);
            ZASSERT_EXIT(bOK);

			if (L.length() != (1 + 2 * n))
			{
				ZASSERT_EXIT(false);
			}
			
			for (int i = 1; i < L.length(); i += 2)
			{
				QPointF pt;
				pt.setX(L[i].toFloat(&bOK));
				ZASSERT_EXIT(bOK);

				pt.setY(L[i + 1].toFloat(&bOK));
				ZASSERT_EXIT(bOK);

				points.append(pt);
			}

			L = handlersStr.split(" ");
			for (int i = 0; i < L.length(); i += 2)
			{
				QPointF pt;

				pt.setX(L[i].toFloat(&bOK));
				ZASSERT_EXIT(bOK);

				pt.setY(L[i + 1].toFloat(&bOK));
				ZASSERT_EXIT(bOK);

				handlers.append(pt);
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

        CurveModel *pModel = new CurveModel("Channel-X", rg, this);
		pModel->initItems(rg, points, handlers);
		pEditor->addCurve(pModel);

		CurveModel *pModel2 = new CurveModel("Channel-Y", rg, this);
        pModel2->initItems({{rg.xFrom, rg.yTo}, {rg.xTo, rg.yFrom}}, {{0,0}, {0,0}, {0,0}, {0,0}});
		pEditor->addCurve(pModel2);

		CurveModel *pModel3 = new CurveModel("Channel-Z", rg, this);
        pModel3->initItems({{rg.xFrom, rg.yTo / 2}, {rg.xTo, rg.yTo / 2}}, {{0,0}, {0,0}, {0,0}, {0,0}});
		pEditor->addCurve(pModel3);

		pEditor->show();
	}
}
#endif
