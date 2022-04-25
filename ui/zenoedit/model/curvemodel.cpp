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

bool CurveModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    int r = index.row(), n = rowCount();
    switch (role)
    {
        case ROLE_NODEPOS:
        {
            QPointF newNodePos = value.toPointF();
            if (r == 0)
            {
                newNodePos.setX(m_range.xFrom);
            }
            else if (r == n - 1)
            {
                newNodePos.setX(m_range.xTo);
            }
            else
            {
                QPointF lastPos = this->data(this->index(r - 1, 0), ROLE_NODEPOS).toPointF();
                QPointF nextPos = this->data(this->index(r + 1, 0), ROLE_NODEPOS).toPointF();
                newNodePos.setX(qMin(qMax(newNodePos.x(), lastPos.x()), nextPos.x()));
            }
            newNodePos.setY(qMin(qMax(newNodePos.y(), m_range.yFrom), m_range.yTo));
            QStandardItemModel::setData(index, newNodePos, role);

            //adjust handles
            if (r > 0 && r < n - 1)
            {
                QPointF lastPos = this->data(this->index(r - 1, 0), ROLE_NODEPOS).toPointF();
                QPointF nextPos = this->data(this->index(r + 1, 0), ROLE_NODEPOS).toPointF();
                QPointF lastRightHdl = this->data(this->index(r - 1, 0), ROLE_RIGHTPOS).toPointF();
                
                QPointF realPos = lastPos + lastRightHdl;
                if (realPos.x() > newNodePos.x())
                {
                    lastRightHdl.setX(newNodePos.x() - lastPos.x());
                    QStandardItemModel::setData(this->index(r - 1, 0), lastRightHdl, ROLE_RIGHTPOS);
                }

                QPointF nextLeftHdl = this->data(this->index(r + 1, 0), ROLE_LEFTPOS).toPointF();
                realPos = nextPos + nextLeftHdl;
                if (realPos.x() < newNodePos.x())
                {
                    nextLeftHdl.setX(newNodePos.x() - nextPos.x());
                    QStandardItemModel::setData(this->index(r + 1, 0), nextLeftHdl, ROLE_LEFTPOS);
                }
            }
            return true;
        }
        case ROLE_LEFTPOS:
        {
            if (r == 0)
                return false;

            QPointF oldPos = this->data(index, ROLE_LEFTPOS).toPointF();
            if (oldPos == value.toPointF())
                return false;

            QPointF lastPos = this->data(this->index(r - 1, 0), ROLE_NODEPOS).toPointF();
            QPointF nodePos = this->data(index, ROLE_NODEPOS).toPointF();
            QPointF realPos = nodePos + value.toPointF();
            realPos.setX(qMin(qMax(realPos.x(), lastPos.x()), nodePos.x()));

            QPointF offset = realPos - nodePos;
            QStandardItemModel::setData(index, offset, ROLE_LEFTPOS);

            //update the other side handle.
            HANDLE_TYPE nodeType = (HANDLE_TYPE)this->data(this->index(r, 0), ROLE_TYPE).toInt();
            QPointF leftOffset = this->data(index, ROLE_LEFTPOS).toPointF();
            QPointF rightOffset = this->data(index, ROLE_RIGHTPOS).toPointF();
            if (nodeType != HDL_FREE && rightOffset != QPointF(0, 0))
            {
                QVector2D roffset(rightOffset);
                QVector2D loffset(leftOffset);
                qreal length = roffset.length();
                if (nodeType == HDL_ALIGNED)
                    length = loffset.length();
                roffset = -loffset.normalized() * length;

                /*QStandardItemModel::*/setData(index, roffset.toPointF(), ROLE_RIGHTPOS);
            }
            return true;
        }
        case ROLE_RIGHTPOS:
        {
            if (r == n - 1)
                return false;

            QPointF oldPos = this->data(index, ROLE_RIGHTPOS).toPointF();
            if (oldPos == value.toPointF())
                return false;

            QPointF nextPos = this->data(this->index(r + 1, 0), ROLE_NODEPOS).toPointF();
            QPointF nodePos = this->data(this->index(r, 0), ROLE_NODEPOS).toPointF();
            QPointF realPos = nodePos + value.toPointF();
            realPos.setX(qMin(qMax(realPos.x(), nodePos.x()), nextPos.x()));

            QPointF offset = realPos - nodePos;
            QStandardItemModel::setData(this->index(r, 0), offset, ROLE_RIGHTPOS);

            //update the other side handle.
            HANDLE_TYPE nodeType = (HANDLE_TYPE)this->data(this->index(r, 0), ROLE_TYPE).toInt();
            QPointF leftOffset = this->data(index, ROLE_LEFTPOS).toPointF();
            QPointF rightOffset = this->data(index, ROLE_RIGHTPOS).toPointF();

            if (nodeType != HDL_FREE && leftOffset != QPointF(0, 0))
            {
                QVector2D roffset(rightOffset);
                QVector2D loffset(leftOffset);
                qreal length = loffset.length();
                if (nodeType == HDL_ALIGNED)
                    length = roffset.length();
                loffset = -roffset.normalized() * length;

                /*QStandardItemModel::*/setData(index, loffset.toPointF(), ROLE_LEFTPOS);
            }
            return true;
        }
        case ROLE_TYPE:
        {
            //todo:
            break;
        }
    }
    return QStandardItemModel::setData(index, value, role);
}