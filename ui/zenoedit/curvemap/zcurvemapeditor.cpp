#include "zcurvemapeditor.h"
#include "curvemapview.h"


ZCurveMapEditor::ZCurveMapEditor(QWidget* parent)
	: QWidget(parent)
	, m_view(nullptr)
{

}

ZCurveMapEditor::~ZCurveMapEditor()
{

}

void ZCurveMapEditor::init(CURVE_RANGE range, const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
	QVBoxLayout* pLayout = new QVBoxLayout;
	pLayout->setContentsMargins(0, 0, 0, 0);
	m_view = new CurveMapView;
	m_view->init(range, pts, handlers);
	pLayout->addWidget(m_view);
	setLayout(pLayout);
}