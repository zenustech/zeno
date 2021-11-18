#include "framework.h"
#include "nodescene.h"
#include "zenonode.h"
#include "ztfutil.h"
#include "resizerectitem.h"
#include "timelineitem.h"


NodeScene::NodeScene(QObject* parent)
	: QGraphicsScene(parent)
	, m_nLargeCellRows(10)
	, m_nLargeCellColumns(12)
	, m_nCellsInLargeCells(5)
	, m_nPixelsInCell(12)
	, m_pHTimeline(nullptr)
{
	//initGrid();
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

void NodeScene::initTimelines()
{
	m_pHTimeline = new TimelineItem;
	addItem(m_pHTimeline);
	connect(this, SIGNAL(changed(QList<QRectF>)), SLOT(timelineChanged()));
}

void NodeScene::timelineChanged()
{
	QList<QGraphicsView*> views = this->views();
	bool empty = views.isEmpty();
	for (int i = 0; i < views.length(); i++)
	{
		QGraphicsView* pView = views[i];
		if (pView)
		{
			QRect rcViewport = pView->viewport()->rect();
			m_pHTimeline->setPos(pView->mapToScene(rcViewport.topLeft()));
		}
	}
}

void NodeScene::initNode()
{
	ZenoNode* pNode = new ZenoNode(this);
	pNode->initStyle(m_nodeparam);
}