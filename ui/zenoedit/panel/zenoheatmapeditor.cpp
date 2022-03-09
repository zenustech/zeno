#include "ui_zenoheatmapeditor.h"
#include "zenoheatmapeditor.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/nodesys/zenosvgitem.h>


ZenoRampSelector::ZenoRampSelector(const QColor& clr, int y, QGraphicsItem* parent)
	: _base(parent)
	, m_color(clr)
{
	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);
	setRect(0, 0, m_size, m_size);
	QColor borderClr(0, 0, 0);
	QPen pen(borderClr, 2);
	pen.setJoinStyle(Qt::MiterJoin);
	setPen(pen);
	setBrush(m_color);

	m_y = y - m_size / 2;
	setPos(0, m_y);
}

void ZenoRampSelector::setColor(const QColor& clr)
{
	m_color = clr;
	setBrush(m_color);
}

void ZenoRampSelector::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	Q_UNUSED(widget);
	painter->setPen(this->pen());
	painter->setBrush(this->brush());
	if ((this->spanAngle() != 0) && (qAbs((this->spanAngle()) % (360 * 16) == 0)))
		painter->drawEllipse(rect());
	else
		painter->drawPie(rect(), startAngle(), spanAngle());
}

QVariant ZenoRampSelector::itemChange(GraphicsItemChange change, const QVariant& value)
{
	if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		bool bSelected = isSelected();
		if (bSelected)
		{
			QPen pen(QColor(255, 128, 0), 2);
			setPen(pen);
		}
		else
		{
			QColor borderClr(0, 0, 0);
			QPen pen(borderClr, 2);
			setPen(pen);
		}
	}
	else if (change == QGraphicsItem::ItemPositionHasChanged)
	{
		int x = pos().x();
		x = qMax(0, x);
		QPointF fixPos(pos().x(), m_y);
		setPos(QPointF(x, m_y));
	}
	return _base::itemChange(change, value);
}


ZenoRampBar::ZenoRampBar(QWidget* parent)
	: QGraphicsView(parent)
{
	static const int barHeight = ZenoStyle::dpiScaled(32);
	setFixedHeight(barHeight);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	m_scene = new QGraphicsScene;
	QGraphicsRectItem* pColorItem = new QGraphicsRectItem(0, 0, 272, barHeight);

	QLinearGradient initBg(0, 0, 272, 0);
	initBg.setColorAt(0, QColor("#5338B0"));
	initBg.setColorAt(1, QColor("#12D7F6"));
	QBrush brush(initBg);
	pColorItem->setBrush(brush);

	QGraphicsLineItem* pLineItem = new QGraphicsLineItem;
	int y = barHeight / 2;
	pLineItem->setLine(0, y, 272, y);

	ZenoRampSelector* selector = new ZenoRampSelector(QColor("#5338B0"), y);

	m_scene->addItem(pColorItem);
	m_scene->addItem(pLineItem);
	m_scene->addItem(selector);

	setScene(m_scene);
}

ZenoHeatMapEditor::ZenoHeatMapEditor(QWidget* parent)
	: QWidget(parent)
{
	setWindowFlags(Qt::Dialog | Qt::MSWindowsFixedSizeDialogHint);
	m_ui = new Ui::HeatMapEditor;
	m_ui->setupUi(this);
	init();
	initSignals();
	installFilters();
}

void ZenoHeatMapEditor::init()
{
	initRamps();
	m_ui->cbPreset->addItems({"BlackBody", "Grayscale", "InfraRed", "TwoTone", "WhiteToRed"});
	m_ui->hueSlider;
}

void ZenoHeatMapEditor::installFilters()
{
}

void ZenoHeatMapEditor::initSignals()
{
}

void ZenoHeatMapEditor::initRamps()
{
}

bool ZenoHeatMapEditor::eventFilter(QObject* watched, QEvent* event)
{
	return QWidget::eventFilter(watched, event);
}

void ZenoHeatMapEditor::dragEnterEvent(QDragEnterEvent* event)
{
	if (event->mimeData()->hasFormat("Label"))
		event->acceptProposedAction();
}

void ZenoHeatMapEditor::mousePressEvent(QMouseEvent* event)
{
	QWidget::mousePressEvent(event);
	QWidget* child = childAt(event->pos());
	if (qobject_cast<QLabel*>(child))
		createDrag(event->pos(), child);
}

void ZenoHeatMapEditor::createDrag(const QPoint& pos, QWidget* widget)
{
	if (widget == Q_NULLPTR)
		return;
	QByteArray byteArray(reinterpret_cast<char*>(&widget), sizeof(QWidget*));
	QDrag* drag = new QDrag(this);
	QMimeData* mimeData = new QMimeData;
	mimeData->setData("Label", byteArray);
	drag->setMimeData(mimeData);
	QPoint globalPos = mapToGlobal(pos);
	QPoint p = widget->mapFromGlobal(globalPos);
	drag->setHotSpot(p);
	drag->setPixmap(widget->grab());
	drag->exec(Qt::CopyAction | Qt::MoveAction);
}

void ZenoHeatMapEditor::dropEvent(QDropEvent* event)
{
	QWidget::dropEvent(event);
}