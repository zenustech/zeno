#include "heatmapnode.h"
#include "panel/zenoheatmapeditor.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"


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
	if (param.control == CONTROL_HEATMAP)
	{
		ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("color", m_renderParams.paramFont, m_renderParams.paramClr.color());
		pParamLayout->addItem(pNameItem);

		ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
		pParamLayout->addItem(pEditBtn);
		connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));
	}
	else
	{
		ZenoNode::initParam(ctrl, pParamLayout, name, param);
	}
}

void MakeHeatMapNode::onEditClicked()
{
	PARAMS_INFO params = index().data(ROLE_PARAMETERS).value<PARAMS_INFO>();
	if (params.find("color") != params.end())
	{
		const QLinearGradient& grad = params["color"].value.value<QLinearGradient>();
		ZenoHeatMapEditor* editor = new ZenoHeatMapEditor(grad);
		int ret = editor->exec();
		//COLOR_RAMPS newRamps = editor->colorRamps();
		IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
		//pModel->setData2(subGraphIndex(), index(), QVariant::fromValue(newRamps), ROLE_COLORRAMPS);
		editor->deleteLater();
	}
}