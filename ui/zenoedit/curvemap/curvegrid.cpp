#include "curvegrid.h"
#include "curvemapview.h"
#include "curvenodeitem.h"
#include "curveutil.h"
#include <zenoui/util/uihelper.h>
#include "curvesitem.h"
#include "util/log.h"

using namespace curve_util;

CurveGrid::CurveGrid(CurveMapView* pView, const QRectF& rc, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_view(pView)
	, m_initRc(rc)
	, m_bFCurve(true)
{
    QMargins margins = m_view->margins();
    m_initRc = m_initRc.marginsRemoved(margins);
	resetTransform(m_initRc, m_view->range());
}

void CurveGrid::resetTransform(QRectF rc, CURVE_RANGE rg)
{
    QMargins margins = m_view->margins();
    m_initRc = rc;

	//setup the transform.
	QPolygonF polygonIn;
	polygonIn << rc.topLeft()
		<< rc.topRight()
		<< rc.bottomRight()
		<< rc.bottomLeft();

	QPolygonF polygonOut;
	polygonOut << QPointF(rg.xFrom, rg.yTo)
		<< QPointF(rg.xTo, rg.yTo)
		<< QPointF(rg.xTo, rg.yFrom)
		<< QPointF(rg.xFrom, rg.yFrom);

	bool isOk = QTransform::quadToQuad(polygonIn, polygonOut, m_transform);
    ZASSERT_EXIT(isOk);

	m_invTrans = m_transform.inverted(&isOk);
    if (!isOk) {
        zeno::log_warn("cannot invert transform (divide by zero)");
        return;
    }

	//example:
    /*
	QPointF pos1 = m_transform.map(m_initRc.topLeft());
	QPointF pos2 = m_transform.map(m_initRc.bottomRight());
	QPointF pos3 = m_transform.map(m_initRc.center());
	*/
}

void CurveGrid::addCurve(CurveModel* model)
{
    CurvesItem* pCurves = new CurvesItem(m_view, this, m_initRc, this);
	pCurves->initCurves(model);
    QString id = model->id();
    ZASSERT_EXIT(!id.isEmpty());
    m_curves[id] = pCurves;
}

QPointF CurveGrid::logicToScene(QPointF logicPos)
{
    QPointF pos = m_invTrans.map(logicPos);
    return pos;
}

QPointF CurveGrid::sceneToLogic(QPointF scenePos)
{
    QPointF pos = m_transform.map(scenePos);
    return pos;
}

bool CurveGrid::isFuncCurve() const
{
    return m_bFCurve;
}

void CurveGrid::setCurvesVisible(QString id, bool bVisible)
{
    if (m_curves.find(id) != m_curves.end())
	{
        m_curves[id]->_setVisible(bVisible);
	}
}

void CurveGrid::setCurvesColor(QString id, QColor color)
{
    if (m_curves.find(id) != m_curves.end())
	{
        m_curves[id]->setColor(color);
    }
}

void CurveGrid::setColor(const QColor& clrGrid, const QColor& clrBackground)
{
	m_clrGrid = clrGrid;
	m_clrBg = clrBackground;
}

QRectF CurveGrid::boundingRect() const
{
	return m_initRc;
}

void CurveGrid::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	QGraphicsObject::mousePressEvent(event);
	QPointF pos = event->pos();
	QPointF wtf = mapFromScene(pos);
	QPointF pos2 = m_transform.map(pos);
}

void CurveGrid::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	QVarLengthArray<QLineF, 256> innerLines;

	const QRectF& rc = boundingRect();

	int W = rc.width(), H = rc.height();
	qreal factor = m_view->factor();

	const QTransform& trans = m_view->transform();
	auto frames = curve_util::numframes(trans.m11(), trans.m22());
    int nVLines = frames.first;
	int nHLines = frames.second;

	const qreal left = rc.left(), right = rc.right();
	const qreal top = rc.top(), bottom = rc.bottom();

	for (qreal x = left; x < right;)
	{
		innerLines.append(QLineF(x, top, x, bottom));
		qreal steps = qMax(1., (right - left) / nVLines);
		x += steps;
	}

	for (qreal y = top; y < bottom;)
	{
		innerLines.append(QLineF(left, y, right, y));
		qreal steps = qMax(1., (bottom - top) / nHLines);
		y += steps;
	}

	painter->save();
	painter->setRenderHint(QPainter::Antialiasing, false);
	painter->fillRect(rc, m_clrBg);
	
	QPen pen(QColor(m_clrGrid), 1. / factor);
	painter->setPen(pen);

	painter->drawRect(rc);
	painter->drawLines(innerLines.data(), innerLines.size());
    painter->restore();
}
