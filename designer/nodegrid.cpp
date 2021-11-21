#include "framework.h"
#include "nodegrid.h"


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


NodeGridItem::NodeGridItem(QGraphicsItem* parent)
    : QGraphicsRectItem(parent)
    , m_nLargeCellRows(10)
    , m_nLargeCellColumns(12)
    , m_nCellsInLargeCells(5)
    , m_nPixelsInCell(9)
    , m_factor(1)
{
    int W = m_nLargeCellColumns * m_nCellsInLargeCells * m_nPixelsInCell;
    int H = m_nLargeCellRows * m_nCellsInLargeCells * m_nPixelsInCell;
    setRect(0, 0, W, H);

    //fill background
    setBrush(QColor(96, 96, 96));
    setPen(QPen(Qt::NoPen));

    QColor clrSmall(104, 104, 104);
    QColor clrBig(184, 184, 184);

    for (int c = 0; c < m_nLargeCellColumns; c++)
    {
        int x = c * m_nCellsInLargeCells * m_nPixelsInCell;
        int yend = H;

        NodeGridLineItem* item = new NodeGridLineItem(this, x, 0, x, yend, this);
        item->setPen(QPen(QColor(81, 87, 92)));

        for (int n = 0; n <= m_nCellsInLargeCells; n++)
        {
            QColor lineColor;
            if (n == 0 || n == m_nCellsInLargeCells)
            {
                lineColor = clrBig;
            }
            else
            {
                lineColor = clrSmall;
            }

            item = new NodeGridLineItem(this, x + n * m_nPixelsInCell, 0, x + n * m_nPixelsInCell, yend, this);
            item->setPen(QPen(lineColor));
        }
    }

    for (int r = 0; r < m_nLargeCellRows; r++)
    {
        int y = r * m_nCellsInLargeCells * m_nPixelsInCell;
        int xend = W;

        NodeGridLineItem* item = new NodeGridLineItem(this, 0, y, xend, y, this);
        item->setPen(QPen(QColor(81, 87, 92)));

        for (int n = 0; n <= m_nCellsInLargeCells; n++)
        {
            QColor lineColor;
            if (n == 0 || n == m_nCellsInLargeCells)
            {
                lineColor = clrBig;
            }
            else
            {
                lineColor = clrSmall;
            }

            item = new NodeGridLineItem(this, 0, y + n * m_nPixelsInCell, xend, y + n * m_nPixelsInCell, this);
            item->setPen(QPen(lineColor));
        }
    }
}