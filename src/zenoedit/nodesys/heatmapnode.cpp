#include "heatmapnode.h"

MakeHeatMapNode::MakeHeatMapNode(const NodeUtilParam& params, QGraphicsItem* parent)
	: ZenoNode(params, parent)
{

}

MakeHeatMapNode::~MakeHeatMapNode()
{

}

QGraphicsLayout* MakeHeatMapNode::initParams()
{
	return ZenoNode::initParams();
}

void MakeHeatMapNode::initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& name, const PARAM_INFO& param)
{
	ZenoNode::initParam(ctrl, pParamLayout, name, param);
}

QGraphicsLinearLayout* MakeHeatMapNode::initCustomParamWidgets()
{
	QGraphicsLinearLayout* pParamLayout = new QGraphicsLinearLayout(Qt::Horizontal);
	ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("color", m_renderParams.paramFont, m_renderParams.paramClr.color());
	pParamLayout->addItem(pNameItem);

	ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
	pParamLayout->addItem(pEditBtn);

	return pParamLayout;
}