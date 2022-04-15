#include "curvemodel.h"


CurveModel::CurveModel(const CURVE_RANGE& rg, QObject* parent)
    : QStandardItemModel(parent)
    , m_range(rg)
{

}

CurveModel::CurveModel(const CURVE_RANGE& rg, int rows, int columns, QObject *parent)
    : QStandardItemModel(rows, columns, parent)
    , m_range(rg)
{

}

CurveModel::~CurveModel()
{

}

void CurveModel::initItems(CURVE_RANGE rg, const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
    m_range = rg;
    int N = pts.size();
    Q_ASSERT(N * 2 == handlers.size());
    for (int i = 0; i < N; i++)
    {
        QPointF logicPos = pts[i];
        QPointF leftOffset = handlers[i * 2];
        QPointF rightOffset = handlers[i * 2 + 1];

        QStandardItem* pItem = new QStandardItem;
        pItem->setData(logicPos, ROLE_NODEPOS);
        pItem->setData(leftOffset, ROLE_LEFTPOS);
        pItem->setData(rightOffset, ROLE_RIGHTPOS);
        pItem->setData(HDL_ASYM, ROLE_TYPE);
        appendRow(pItem);
    }
}

void CurveModel::resetRange(const CURVE_RANGE& rg)
{
    m_range = rg;
    emit rangeChanged(m_range);
}

CURVE_RANGE CurveModel::range() const
{
    return m_range;
}