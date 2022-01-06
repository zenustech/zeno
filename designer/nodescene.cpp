#include "framework.h"
#include "nodescene.h"
#include "nodetemplate.h"
#include <render/ztfutil.h>
#include "resizableitemimpl.h"
#include "componentitem.h"
#include "timelineitem.h"
#include "dragpointitem.h"
#include "nodegrid.h"
#include "nodesview.h"
#include "styletabwidget.h"
#include "nodeswidget.h"
#include "designermainwin.h"
#include "util.h"


NodeScene::NodeScene(NodesView* pView, QObject* parent)
	: QGraphicsScene(parent)
	, m_nLargeCellRows(10)
	, m_nLargeCellColumns(12)
	, m_nCellsInLargeCells(5)
	, m_nPixelsInCell(PIXELS_IN_CELL)
	, m_pHTimeline(nullptr)
	, m_selectedRect(nullptr)
	, m_selectedItem(nullptr)
	, m_grid(nullptr)
	, m_pNode(nullptr)
{
}

NodeScene::~NodeScene()
{
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

void NodeScene::resetPreset(int W, int H)
{
    if (m_grid)
		delete m_grid;
    m_grid = new NodeGridItem(QSize(W, H));
    addItem(m_grid);
    update();
}

void NodeScene::updateTimeline(qreal factor)
{
	if (m_pHTimeline)
		m_pHTimeline->updateScalar(factor);
	if (m_pVTimeline)
		m_pVTimeline->updateScalar(factor);
}

QSizeF NodeScene::getSceneSize()
{
    return m_grid->boundingRect().size();
}

void NodeScene::initNode()
{
	if (m_pNode == nullptr)
		m_pNode = new NodeTemplate(this);

	m_pNode->initStyleModel(m_nodeparam);
    connect(m_pNode, SIGNAL(markDirty()), this, SIGNAL(markDirty()));
    connect(this, &NodeScene::markDirty, [=]() {
        getMainWindow()->getCurrentTab()->markDirty(true);
    });
}

QStandardItemModel* NodeScene::model() const
{
    return m_pNode ? m_pNode->model() : nullptr;
}

QItemSelectionModel* NodeScene::selectionModel() const
{
    return m_pNode ? m_pNode->selectionModel() : nullptr;
}

NodeParam NodeScene::exportNodeParam()
{
    return m_pNode->exportParam();
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