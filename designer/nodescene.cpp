#include "framework.h"
#include "nodescene.h"
#include "zenonode.h"
#include "ztfutil.h"
#include "resizerectitem.h"


NodeScene::NodeScene(QObject* parent)
	: QGraphicsScene(parent)
	, m_nLargeCellRows(10)
	, m_nLargeCellColumns(12)
	, m_nCellsInLargeCells(5)
	, m_nPixelsInCell(12)
{
	initGrid();
}

void NodeScene::initSkin(const QString& fn)
{
	m_nodeparam = ZtfUtil::GetInstance().loadZtf(fn);
}

void NodeScene::initGrid()
{
	int W = m_nLargeCellColumns * m_nCellsInLargeCells * m_nPixelsInCell;
	int H = m_nLargeCellRows * m_nCellsInLargeCells * m_nPixelsInCell;

	//fill background
	addRect(QRectF(0, 0, W, H), QPen(Qt::NoPen), QColor(51, 59, 62));

	for (int r = 0; r < m_nLargeCellRows; r++)
	{
		int y = r * m_nCellsInLargeCells * m_nPixelsInCell;
		int xend = W;
		addLine(0, y, xend, y, QPen(QColor(81, 87, 92)));
		for (int n = 0; n <= m_nCellsInLargeCells; n++)
		{
			QColor lineColor;
			if (n == 0 || n == m_nCellsInLargeCells)
			{
				lineColor = QColor(81, 87, 92);
			}
			else
			{
				lineColor = QColor(58, 65, 71);
			}
			addLine(0, y + n * m_nPixelsInCell, xend, y + n * m_nPixelsInCell, QPen(lineColor));
		}
	}

	for (int c = 0; c < m_nLargeCellColumns; c++)
	{
		int x = c * m_nCellsInLargeCells * m_nPixelsInCell;
		int yend = H;
		addLine(x, 0, x, yend, QPen(QColor(81, 87, 92)));
		for (int n = 0; n <= m_nCellsInLargeCells; n++)
		{
			QColor lineColor;
			if (n == 0 || n == m_nCellsInLargeCells)
			{
				lineColor = QColor(81, 87, 92);
			}
			else
			{
				lineColor = QColor(58, 65, 71);
			}
			addLine(x + n * m_nPixelsInCell, 0, x + n * m_nPixelsInCell, yend, QPen(lineColor));
		}
	}
}

void NodeScene::initNode()
{
	//ZenoNode* pNode = new ZenoNode(this);
	//pNode->initStyle(m_nodeparam);

	//ScreenGrabRect* pNode = new ScreenGrabRect(QRectF(50, 50, 100, 30));
	//addItem(pNode);
	//pNode->installEventFilter(this);

	//ResizableComponentItem* ptemp = new ResizableComponentItem(50, 40, 100, 30);
	//addItem(ptemp);

	ResizableRectItem* ptemp2 = new ResizableRectItem(150, 140, 100, 30);
	addItem(ptemp2);
	//QPointF pos = ptemp->pos();
	//QPointF scenePos = ptemp->scenePos();

	return;

	//auto m_ltcorner = new QGraphicsRectItem(0, 0, 8, 8);
	//m_ltcorner->setPen(QPen(QColor(21, 152, 255), 1));
	//m_ltcorner->setBrush(QBrush(QColor(255, 255, 255)));
	//m_ltcorner->setZValue(100);
	//m_ltcorner->setPos(QPointF(0, 0));
	//addItem(m_ltcorner);
}