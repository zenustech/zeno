#include "nodegrid.h"
#include "nodesys_common.h"


NodeGridLineItem::NodeGridLineItem(NodeGridItem* grid, qreal x1, qreal y1, qreal x2, qreal y2, QGraphicsItem* parent)
    : QGraphicsLineItem(x1, y1, x2, y2, parent)
    , m_grid(grid)
{
}

void NodeGridLineItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    qreal factor = m_grid->factor();
    QPen pen = this->pen();
    pen.setWidthF(pen.widthF() / factor);
    painter->setPen(pen);
    painter->drawLine(line());
}


NodeGridItem::NodeGridItem(QSize sz, QGraphicsItem *parent)
    : QGraphicsRectItem(parent)
    , m_nCellsInLargeCells(5)
    , m_nPixelsInCell(PIXELS_IN_CELL)
    , m_factor(1)
{
    m_nLargeCellColumns = sz.width() / (m_nCellsInLargeCells * m_nPixelsInCell);
    m_nLargeCellRows = sz.height() / (m_nCellsInLargeCells * m_nPixelsInCell);
    int W = m_nLargeCellColumns * m_nCellsInLargeCells * m_nPixelsInCell;
    int H = m_nLargeCellRows * m_nCellsInLargeCells * m_nPixelsInCell;
    initGrid(W, H);
}

NodeGridItem::NodeGridItem(QGraphicsItem *parent)
    : QGraphicsRectItem(parent)
    , m_nLargeCellRows(10)
    , m_nLargeCellColumns(12)
    , m_nCellsInLargeCells(5)
    , m_nPixelsInCell(PIXELS_IN_CELL)
    , m_factor(1)
{
    int W = m_nLargeCellColumns * m_nCellsInLargeCells * m_nPixelsInCell;
    int H = m_nLargeCellRows * m_nCellsInLargeCells * m_nPixelsInCell;
    initGrid(W, H);
}

void NodeGridItem::initGrid(int W, int H)
{
    setRect(0, 0, W, H);
    setZValue(ZVALUE_GRID_BACKGROUND);

    //fill background
    setBrush(QColor(96, 96, 96));
    setPen(QPen(Qt::NoPen));

    QColor clrSmall(104, 104, 104);
    QColor clrBig(184, 184, 184);

    for (int c = 0; c < m_nLargeCellColumns; c++)
    {
        int x = c * m_nCellsInLargeCells * m_nPixelsInCell;
        int yend = H;

        for (int n = 0; n <= m_nCellsInLargeCells; n++)
        {
            QColor lineColor;
            int depth = 0;
            if (n == 0 || n == m_nCellsInLargeCells)
            {
                lineColor = clrBig;
                depth = ZVALUE_GRID_BIG;
            }
            else
            {
                lineColor = clrSmall;
                depth = ZVALUE_GRID_SMALL;
            }

            auto item = new NodeGridLineItem(this, x + n * m_nPixelsInCell, 0, x + n * m_nPixelsInCell, yend, this);
            item->setZValue(depth);
            item->setPen(QPen(lineColor));
        }
    }

    for (int r = 0; r < m_nLargeCellRows; r++)
    {
        int y = r * m_nCellsInLargeCells * m_nPixelsInCell;
        int xend = W;

        for (int n = 0; n <= m_nCellsInLargeCells; n++)
        {
            QColor lineColor;
            int depth = 0;
            if (n == 0 || n == m_nCellsInLargeCells)
            {
                lineColor = clrBig;
                depth = ZVALUE_GRID_BIG;
            }
            else
            {
                lineColor = clrSmall;
                depth = ZVALUE_GRID_SMALL;
            }

            auto item = new NodeGridLineItem(this, 0, y + n * m_nPixelsInCell, xend, y + n * m_nPixelsInCell, this);
            item->setPen(QPen(lineColor));
            item->setZValue(depth);
        }
    }
}