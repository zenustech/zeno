#include "framework.h"
#include "nodescene.h"
#include "zenonode.h"
#include "ztfutil.h"
#include "resizableitemimpl.h"
#include "componentitem.h"
#include "timelineitem.h"
#include "dragpointitem.h"
#include "nodegrid.h"
#include "nodesview.h"


NodeScene::NodeScene(NodesView* pView, QObject* parent)
	: QGraphicsScene(parent)
	, m_nLargeCellRows(10)
	, m_nLargeCellColumns(12)
	, m_nCellsInLargeCells(5)
	, m_nPixelsInCell(12)
	, m_pHTimeline(nullptr)
	, m_selectedRect(nullptr)
	, m_selectedItem(nullptr)
	, m_grid(nullptr)
	, m_model(new QStandardItemModel(pView))
	, m_selection(new QItemSelectionModel(m_model))
	, m_pNode(nullptr)
{
	m_model->setObjectName(NODE_MODEL_NAME);
	m_selection->setObjectName(NODE_SELECTION_MODEL);
	connect(this, SIGNAL(selectionChanged()), this, SLOT(onSelectionChanged()));
}

NodeScene::~NodeScene()
{
	m_model->destroyed();
	m_selection->destroyed();
	m_model = nullptr;
	m_selection = nullptr;
}

void NodeScene::initSkin(const QString& fn)
{
	m_nodeparam = ZtfUtil::GetInstance().loadZtf(fn);
}

void NodeScene::initGrid()
{
	m_grid = new NodeGridItem;
	addItem(m_grid);
}

void NodeScene::initTimelines(QRectF rcView)
{
    m_pHTimeline = new TimelineItem(this, true, rcView);
    addItem(m_pHTimeline);
    m_pVTimeline = new TimelineItem(this, false, rcView);
    addItem(m_pVTimeline);
}

void NodeScene::onViewTransformChanged(qreal factor)
{
	updateTimeline(factor);
	m_grid->setFactor(factor);
}

void NodeScene::updateTimeline(qreal factor)
{
	if (m_pHTimeline)
		m_pHTimeline->updateScalar(factor);
	if (m_pVTimeline)
		m_pVTimeline->updateScalar(factor);
}

void NodeScene::initNode()
{
	if (m_pNode == nullptr)
		m_pNode = new ZenoNode(this);

	m_pNode->initStyle(m_nodeparam);
	m_pNode->initModel(m_model);

	connect(m_selection, SIGNAL(selectionChanged(const QItemSelection&, const QItemSelection&)),
		m_pNode, SLOT(onSelectionChanged(const QItemSelection&, const QItemSelection&)));

	//ComponentItem* pItem = new ComponentItem(this, 50, 50, 150, 30);
	//ComponentItem* pItem2 = new ComponentItem(this, 350, 350, 150, 30);

	//m_originalPix = QPixmap("C:\\editor\\uirender\\Header_back_board.jpg");
	//m_coreitem = new QGraphicsPixmapItem(m_originalPix.scaled(m_width, m_height), this);

	//ResizablePixmapItem* pItem = new ResizablePixmapItem(this, QPixmap("C:\\editor\\uirender\\view.jpg").scaled(60, 61));
	//pItem->setPos(50, 50);
}

QStandardItemModel* NodeScene::model() const
{
	return m_model;
}

QItemSelectionModel* NodeScene::selectionModel() const
{
	return m_selection;
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