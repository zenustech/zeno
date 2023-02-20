#include "curvemodel.h"
#include <zeno/utils/pybjson.h>
#include <zeno/utils/log.h>


CurveModel::CurveModel(const QString& id, const CURVE_RANGE& rg, QObject* parent)
    : QStandardItemModel(parent)
    , m_range(rg)
    , m_id(id)
{
}

CurveModel::CurveModel(const QString& id, const CURVE_RANGE& rg, int rows, int columns, QObject *parent)
    : QStandardItemModel(rows, columns, parent)
    , m_range(rg)
    , m_id(id)
{
}

CurveModel::~CurveModel()
{
}

CURVE_DATA CurveModel::getItems() const {
    CURVE_DATA dat;
    dat.rg = m_range;
    int count = rowCount();
    dat.points.resize(count);
    for (int i = 0; i < count; i++) {
        auto pItem = item(i);
        auto &pt = dat.points[i];
        pt.point = pItem->data(ROLE_NODEPOS).value<QPointF>();
        pt.leftHandler = pItem->data(ROLE_LEFTPOS).value<QPointF>();
        pt.rightHandler = pItem->data(ROLE_RIGHTPOS).value<QPointF>();
        pt.controlType = 0; // pItem->data(ROLE_TYPE);
    }
    dat.cycleType = 0;
    dat.key = m_id;
    return dat;
}

void CurveModel::initItems(CURVE_DATA const &curvedat)
{
    m_range = curvedat.rg;
    auto &pts = curvedat.points;
    int N = pts.size();
    for (int i = 0; i < N; i++)
    {
        QPointF logicPos = pts[i].point;
        QPointF leftOffset = pts[i].leftHandler;
        QPointF rightOffset = pts[i].rightHandler;

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
    //todo: map every pos from m_range to rg;
    QPolygonF polygonIn;
    polygonIn << QPointF(m_range.xFrom, m_range.yFrom) << QPointF(m_range.xTo, m_range.yFrom)
              << QPointF(m_range.xTo, m_range.yTo) << QPointF(m_range.xFrom, m_range.yTo);

    QPolygonF polygonOut;
    polygonOut << QPointF(rg.xFrom, rg.yFrom) << QPointF(rg.xTo, rg.yFrom)
              << QPointF(rg.xTo, rg.yTo) << QPointF(rg.xFrom, rg.yTo);

    QTransform trans, inv_trans;
    bool isOk = QTransform::quadToQuad(polygonIn, polygonOut, trans);
    Q_ASSERT(isOk);
    if (!isOk)
        return;

    inv_trans = trans.inverted(&isOk);
    if (!isOk) {
        zeno::log_warn("cannot invert transform (divide by zero)");
        return;
    }

    m_range = rg;

    for (int r = 0; r < rowCount(); r++)
    {
        const QModelIndex &idx = index(r, 0);
        QPointF nodePos = idx.data(ROLE_NODEPOS).toPointF();
        QPointF leftPos = idx.data(ROLE_LEFTPOS).toPointF();
        QPointF rightPos = idx.data(ROLE_RIGHTPOS).toPointF();
        QPointF nodePos_ = trans.map(nodePos);
        QPointF leftPos_ = trans.map(nodePos + leftPos) - nodePos_;
        QPointF rightPos_ = trans.map(nodePos + rightPos) - nodePos_;

        // no need to adjust anything because position of all nodes scaled equally.
        QStandardItemModel::setData(idx, nodePos_, ROLE_NODEPOS);
        QStandardItemModel::setData(idx, leftPos_, ROLE_LEFTPOS);
        QStandardItemModel::setData(idx, rightPos_, ROLE_RIGHTPOS);
    }
}

CURVE_RANGE CurveModel::range() const
{
    return m_range;
}

QString CurveModel::id() const
{
    return m_id;
}

void CurveModel::setId(QString id) {
    m_id = id;
}

bool CurveModel::isTimeline() const
{
    return m_bTimeline;
}

void CurveModel::setTimeline(bool bTimeline)
{
    m_bTimeline = bTimeline;
}

QPointF CurveModel::clipNodePos(const QModelIndex& index, const QPointF& currPos)
{
    qreal epsilon = (m_range.xTo - m_range.xFrom) / 1000;
    QPointF newNodePos = currPos;
    QPointF oldPos = index.data(ROLE_NODEPOS).toPointF();
    if (newNodePos == oldPos)
        return newNodePos;

    int r = index.row(), n = rowCount();
    bool bLockX = index.data(ROLE_LOCKX).toBool();
    bool bLockY = index.data(ROLE_LOCKY).toBool();

    if (bLockX)
    {
        newNodePos.setX(oldPos.x());
    }
    else if (r == 0)
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
        newNodePos.setX(qMin(qMax(newNodePos.x() + epsilon, lastPos.x()), nextPos.x() - epsilon));
    }

    if (bLockY)
    {
        newNodePos.setY(oldPos.y());
    }
    else
    {
        newNodePos.setY(qMin(qMax(newNodePos.y(), m_range.yFrom), m_range.yTo));
    }
    return newNodePos;
}

QPointF CurveModel::adjustLastRightHdl(const QModelIndex& currIdx)
{
    //index is current moved node.
    int r = currIdx.row(), n = rowCount();
    if (r == 0)
        return QPointF(0, 0);

    const QModelIndex& lastIdx = this->index(r - 1, 0);
    QPointF currPos = this->data(currIdx, ROLE_NODEPOS).toPointF();
    QPointF lastPos = this->data(lastIdx, ROLE_NODEPOS).toPointF();
    QPointF lastRightHdl = this->data(lastIdx, ROLE_RIGHTPOS).toPointF();
    QPointF lastRightHdlPos = lastPos + lastRightHdl;
    if (lastRightHdl.x() == 0 || lastRightHdlPos.x() <= currPos.x())
        return lastRightHdl;

    qreal k = lastRightHdl.y() / lastRightHdl.x();
    qreal newX = currPos.x();
    qreal newY = lastPos.y() - k * (lastPos.x() - newX);
    return QPointF(newX, newY) - lastPos;
}

QPointF CurveModel::adjustNextLeftHdl(const QModelIndex &currIdx)
{
    int r = currIdx.row(), n = rowCount();
    if (r == n)
        return QPointF(0, 0);

    const QModelIndex& nextIdx = this->index(r + 1, 0);
    QPointF nodePos = this->data(currIdx, ROLE_NODEPOS).toPointF();
    QPointF nextPos = this->data(nextIdx, ROLE_NODEPOS).toPointF();
    QPointF nextLeftHdl = this->data(nextIdx, ROLE_LEFTPOS).toPointF();
    QPointF nextLeftHdlPos = nextLeftHdl + nextPos;
    if (nextLeftHdl.x() == 0 || nextLeftHdlPos.x() >= nodePos.x())
        return nextLeftHdl;

    qreal k = nextLeftHdl.y() / nextLeftHdl.x();
    qreal newX = nodePos.x();
    qreal newY = nextPos.y() - k * (nextPos.x() - newX);
    return QPointF(newX, newY) - nextPos;
}

QPointF CurveModel::adjustCurrLeftHdl(const QModelIndex &currIdx)
{
    int r = currIdx.row(), n = rowCount();
    if (r == 0)
        return QPointF(0, 0);

    QPointF lastPos = this->data(index(r - 1, 0), ROLE_NODEPOS).toPointF();
    QPointF currPos = this->data(currIdx, ROLE_NODEPOS).toPointF();
    QPointF leftHdl = this->data(currIdx, ROLE_LEFTPOS).toPointF();
    QPointF leftHdlPos = leftHdl + currPos;
    if (leftHdl.x() == 0 || leftHdlPos.x() >= lastPos.x())
        return leftHdl;

    qreal k = leftHdl.y() / leftHdl.x();
    qreal newX = lastPos.x();
    qreal newY = currPos.y() - k * (currPos.x() - newX);
    return QPointF(newX, newY) - currPos;
}

QPointF CurveModel::adjustCurrRightHdl(const QModelIndex &currIdx)
{
    int r = currIdx.row(), n = rowCount();
    if (r == n - 1)
        return QPointF(0, 0);

    QPointF nextPos = this->data(index(r + 1, 0), ROLE_NODEPOS).toPointF();
    QPointF currPos = this->data(currIdx, ROLE_NODEPOS).toPointF();
    QPointF rightHdl = this->data(currIdx, ROLE_RIGHTPOS).toPointF();
    QPointF rightHdlPos = rightHdl + currPos;
    if (rightHdl.x() == 0 || rightHdlPos.x() <= nextPos.x())
        return rightHdl;

    qreal k = rightHdl.y() / rightHdl.x();
    qreal newX = nextPos.x();
    qreal newY = currPos.y() - k * (currPos.x() - newX);
    return QPointF(newX, newY) - currPos;
}

QPair<QPointF, QPointF> CurveModel::adjustWhenLeftHdlChanged(const QModelIndex &currIdx, QPointF newPos)
{
    int r = currIdx.row();
    if (r == 0)
        return {QPointF(0, 0), this->data(currIdx, ROLE_RIGHTPOS).toPointF()};
    
    HANDLE_TYPE nodeType = (HANDLE_TYPE)this->data(currIdx, ROLE_TYPE).toInt();
    QModelIndex lastIdx = index(r - 1, 0);
    QPointF nodePos = this->data(currIdx, ROLE_NODEPOS).toPointF();
    QPointF lastPos = this->data(lastIdx, ROLE_NODEPOS).toPointF();
    QPointF leftOffset = newPos;

    leftOffset.setX(qMax(qMin(0., leftOffset.x()), lastPos.x() - nodePos.x()));

    if (r == rowCount() - 1)
        return {leftOffset, QPointF(0, 0)};

    // adjust right handle
    QModelIndex nextIdx = index(r + 1, 0);
    QPointF nextPos = this->data(nextIdx, ROLE_NODEPOS).toPointF();
    QPointF rightOffset = this->data(currIdx, ROLE_RIGHTPOS).toPointF();

    if (nodeType != HDL_FREE && rightOffset != QPointF(0, 0))
    {
        QVector2D roffset(rightOffset);
        QVector2D loffset(leftOffset);
        qreal length = roffset.length();
        if (nodeType == HDL_ALIGNED)
            length = loffset.length();
        roffset = -loffset.normalized() * length;

        rightOffset = roffset.toPointF();
        QPointF rightPos = nodePos + roffset.toPointF();
    }

    return {leftOffset, rightOffset};
}

QPair<QPointF, QPointF> CurveModel::adjustWhenRightHdlChanged(const QModelIndex& currIdx, QPointF newPos)
{
    int r = currIdx.row();
    if (r == rowCount() - 1)
        return {this->data(currIdx, ROLE_LEFTPOS).toPointF(), QPointF(0, 0)};

    HANDLE_TYPE nodeType = (HANDLE_TYPE)this->data(currIdx, ROLE_TYPE).toInt();
    QModelIndex nextIdx = index(r + 1, 0);
    QPointF nodePos = this->data(currIdx, ROLE_NODEPOS).toPointF();
    QPointF nextPos = this->data(nextIdx, ROLE_NODEPOS).toPointF();
    QPointF rightOffset = newPos;

    rightOffset.setX(qMax(qMin(nextPos.x() - nodePos.x(), rightOffset.x()), 0.));

    // adjust left handle
    QModelIndex lastIdx = index(r - 1, 0);
    QPointF lastPos = this->data(lastIdx, ROLE_NODEPOS).toPointF();
    QPointF leftOffset = this->data(currIdx, ROLE_LEFTPOS).toPointF();

    if (nodeType != HDL_FREE && leftOffset != QPointF(0, 0))
    {
        QVector2D roffset(rightOffset);
        QVector2D loffset(leftOffset);
        qreal length = loffset.length();
        if (nodeType == HDL_ALIGNED)
            length = roffset.length();
        loffset = -roffset.normalized() * length;

        leftOffset = loffset.toPointF();
    }

    return {leftOffset, rightOffset};
}


bool CurveModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    int r = index.row(), n = rowCount();
    switch (role)
    {
        case ROLE_NODEPOS:
        {
            qreal epsilon = (m_range.xTo - m_range.xFrom) / 1000;
            QPointF newNodePos = clipNodePos(index, value.toPointF());
            QStandardItemModel::setData(index, newNodePos, role);

            //adjust handles
            QPointF lastRightHdl = adjustLastRightHdl(index);
            QStandardItemModel::setData(this->index(r - 1, 0), lastRightHdl, ROLE_RIGHTPOS);

            QPointF currLeftHdl = adjustCurrLeftHdl(index);
            QStandardItemModel::setData(index, currLeftHdl, ROLE_LEFTPOS);

            QPointF currRightHdl = adjustCurrRightHdl(index);
            QStandardItemModel::setData(index, currRightHdl, ROLE_RIGHTPOS);

            QPointF nextLeftHdl = adjustNextLeftHdl(index);
            QStandardItemModel::setData(this->index(r + 1, 0), nextLeftHdl, ROLE_LEFTPOS);
       
            return true;
        }
        case ROLE_LEFTPOS:
        {
            if (r == 0)
            {
                QStandardItemModel::setData(index, QPointF(0, 0), ROLE_LEFTPOS);
                return false;
            }

            QPointF oldPos = this->data(index, ROLE_LEFTPOS).toPointF();
            if (oldPos == value.toPointF())
                return false;

            auto pospair = adjustWhenLeftHdlChanged(index, value.toPointF());
            QPointF leftOffset = pospair.first;
            QPointF rightOffset = pospair.second;
            QStandardItemModel::setData(index, leftOffset, ROLE_LEFTPOS);
            QStandardItemModel::setData(index, rightOffset, ROLE_RIGHTPOS);
            return true;
        }
        case ROLE_RIGHTPOS:
        {
            if (r == n - 1)
            {
                QStandardItemModel::setData(index, QPointF(0, 0), ROLE_RIGHTPOS);
                return false;
            }

            QPointF oldPos = this->data(index, ROLE_RIGHTPOS).toPointF();
            if (oldPos == value.toPointF())
                return false;

            auto pospair = adjustWhenRightHdlChanged(index, value.toPointF());
            QPointF leftOffset = pospair.first;
            QPointF rightOffset = pospair.second;
            QStandardItemModel::setData(index, rightOffset, ROLE_RIGHTPOS);
            QStandardItemModel::setData(index, leftOffset, ROLE_LEFTPOS);
            return true;
        }
        case ROLE_TYPE:
        {
            //todo:
            CURVE_RANGE rg = range();
            qreal xscale = (rg.xTo - rg.xFrom) / 10.;

            HANDLE_TYPE oldType = (HANDLE_TYPE)this->data(index, ROLE_TYPE).toInt();
            HANDLE_TYPE newType = (HANDLE_TYPE)value.toInt();
            //QStandardItemModel::setData(index, newType, ROLE_TYPE);

            switch (newType)
            {
                case HDL_ALIGNED:
                {
                    if (oldType == HDL_VECTOR)
                    {
                        QPointF leftOffset(-xscale, 0);
                        QPointF rightOffset(xscale, 0);
                        QStandardItemModel::setData(index, leftOffset, ROLE_LEFTPOS);
                        QStandardItemModel::setData(index, rightOffset, ROLE_RIGHTPOS);
                    }
                    QPointF leftOffset = this->data(index, ROLE_LEFTPOS).toPointF();
                    QPointF rightOffset = this->data(index, ROLE_RIGHTPOS).toPointF();
                    if (leftOffset != QPointF(0, 0))
                    {
                        rightOffset = -leftOffset;
                        QStandardItemModel::setData(index, rightOffset, ROLE_RIGHTPOS);
                    }
                    else if (rightOffset != QPointF(0, 0))
                    {
                        leftOffset = -rightOffset;
                        QStandardItemModel::setData(index, leftOffset, ROLE_LEFTPOS);
                    }
                    break;
                }
                case HDL_ASYM:
                {
                    if (oldType == HDL_VECTOR)
                    {
                        QPointF leftOffset(-xscale, 0);
                        QPointF rightOffset(xscale, 0);
                        QStandardItemModel::setData(index, leftOffset, ROLE_LEFTPOS);
                        QStandardItemModel::setData(index, rightOffset, ROLE_RIGHTPOS);
                    }
                    break;
                }
                case HDL_FREE:
                {
                    if (oldType == HDL_VECTOR)
                    {
                        QPointF leftOffset(-xscale, 0);
                        QPointF rightOffset(xscale, 0);
                        QStandardItemModel::setData(index, leftOffset, ROLE_LEFTPOS);
                        QStandardItemModel::setData(index, rightOffset, ROLE_RIGHTPOS);
                    }
                    break;
                }
                case HDL_VECTOR:
                {
                    QStandardItemModel::setData(index, QPointF(0, 0), ROLE_LEFTPOS);
                    QStandardItemModel::setData(index, QPointF(0, 0), ROLE_RIGHTPOS);
                    break;
                }
            }
            break;
        }
    }
    return QStandardItemModel::setData(index, value, role);
}

#define CURVE_MODEL_SERIALIZER \
        .obj() \
        .key("range") \
        .obj() \
        .key("xFrom") \
        .val(m_range.xFrom) \
        .key("xTo") \
        .val(m_range.xTo) \
        .key("yFrom") \
        .val(m_range.yFrom) \
        .key("yTo") \
        .val(m_range.yTo) \
        .eobj() \
        .key("id") \
        .val_f([&] { return m_id.toStdString(); }, [&] (auto &&x) { m_id = QString::fromStdString(x); }) \
        .eobj() \
        /**/

std::string CurveModel::z_serialize() const {
    auto dat = getItems();
    return zeno::pybjsonwriter()
        CURVE_MODEL_SERIALIZER
        .str();
}


void CurveModel::z_deserialize(std::string_view s) {
    auto dat = getItems();
    zeno::pybjsonparser().str(s)
        CURVE_MODEL_SERIALIZER
        ;
}
