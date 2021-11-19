#include "framework.h"
#include "nodescene.h"
#include "zenonode.h"
#include "ztfutil.h"
#include "resizerectitem.h"
#include "componentitem.h"
#include "timelineitem.h"
#include "dragpointitem.h"


NodeScene::NodeScene(QObject* parent)
	: QGraphicsScene(parent)
	, m_nLargeCellRows(10)
	, m_nLargeCellColumns(12)
	, m_nCellsInLargeCells(5)
	, m_nPixelsInCell(12)
	, m_pHTimeline(nullptr)
	, m_selectedRect(nullptr)
	, m_selectedItem(nullptr)
{
	//initGrid();
	connect(this, SIGNAL(selectionChanged()), this, SLOT(onSelectionChanged()));
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
	initSelectionDragBorder();
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
	//ComponentItem* pItem = new ComponentItem(this, 50, 50, 150, 30);

	//ComponentItem* pItem2 = new ComponentItem(this, 350, 350, 150, 30);
}

void NodeScene::initSelectionDragBorder()
{
	m_dragPoints.resize(DRAG_RIGHTBOTTOM + 1);
	for (int i = DRAG_LEFTTOP; i <= DRAG_RIGHTBOTTOM; i++)
	{
        QPen pen(QColor(21, 152, 255), borderW);
        pen.setJoinStyle(Qt::MiterJoin);
        QBrush brush(QColor(255, 255, 255));

		m_dragPoints[i] = new DragPointItem((DRAG_ITEM)i, this, dragW, dragH);
		m_dragPoints[i]->setPen(pen);
		m_dragPoints[i]->setBrush(brush);
		m_dragPoints[i]->hide();
		m_dragPoints[i]->setZValue(100);
		addItem(m_dragPoints[i]);
	}

    QPen pen(QColor(21, 152, 255), borderW);
    pen.setJoinStyle(Qt::MiterJoin);
	m_selectedRect = new QGraphicsRectItem;
	m_selectedRect->setPen(pen);
	m_selectedRect->setBrush(Qt::NoBrush);
	m_selectedRect->hide();
	addItem(m_selectedRect);
}

void NodeScene::onSelectionChanged()
{
	return;
	QList<QGraphicsItem*> selItems = this->selectedItems();
	if (selItems.isEmpty())
	{
		m_selectedRect->hide();
        for (int i = DRAG_LEFTTOP; i <= DRAG_RIGHTBOTTOM; i++)
        {
            m_dragPoints[i]->hide();
        }
	}
	else
	{
		QGraphicsItem* pSel = selItems[0];
		_adjustDragRectPos(pSel);
	}
}

void NodeScene::_adjustDragRectPos(QGraphicsItem* pSel)
{
    QRectF br = pSel->sceneBoundingRect();
    m_selectedRect->setRect(QRectF(0, 0, br.width(), br.height()));
    m_selectedRect->setPos(pSel->pos());
    m_selectedRect->show();

    m_dragPoints[DRAG_LEFTTOP]->setPos(br.topLeft());
    m_dragPoints[DRAG_MIDTOP]->setPos(QPointF(br.center().x(), br.top()));
    m_dragPoints[DRAG_RIGHTTOP]->setPos(QPointF(br.right(), br.top()));

    m_dragPoints[DRAG_LEFTMID]->setPos(QPointF(br.left(), br.center().y()));
    m_dragPoints[DRAG_RIGHTMID]->setPos(QPointF(br.right(), br.center().y()));

    m_dragPoints[DRAG_LEFTBOTTOM]->setPos(QPointF(br.left(), br.bottom()));
    m_dragPoints[DRAG_MIDBOTTOM]->setPos(QPointF(br.center().x(), br.bottom()));
    m_dragPoints[DRAG_RIGHTBOTTOM]->setPos(QPointF(br.right(), br.bottom()));

    for (int i = DRAG_LEFTTOP; i <= DRAG_RIGHTBOTTOM; i++)
    {
        m_dragPoints[i]->show();
    }
}

void NodeScene::updateDragPoints(QGraphicsItem* pDragged, DRAG_ITEM dragWay)
{
	return;
	switch (dragWay)
	{
		case DRAG_LEFTTOP:
		{

			break;
		}
		case DRAG_LEFTMID:
		{
			break;
		}
		case DRAG_LEFTBOTTOM:
		{
			break;
		}
		case DRAG_MIDTOP:
		{
			break;
		}
		case DRAG_MIDBOTTOM:
		{
			break;
		}
		case DRAG_RIGHTTOP:
		{
			break;
		}
		case DRAG_RIGHTMID:
		{
			break;
		}
		case DRAG_RIGHTBOTTOM:
		{
			break;
		}
		case TRANSLATE:
		{
			_adjustDragRectPos(pDragged);
		}
	}
}