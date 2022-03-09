#include "ui_zenoheatmapeditor.h"
#include "zenoheatmapeditor.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/nodesys/zenosvgitem.h>


ZenoRampSelector::ZenoRampSelector(ZenoRampBar* pRampBar, QGraphicsItem* parent)
	: _base(parent)
	, m_rampBar(pRampBar)
{
	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);
	setRect(0, 0, m_size, m_size);
	QPen pen(Qt::black, 1);
	pen.setJoinStyle(Qt::MiterJoin);
	setPen(pen);

	QGraphicsDropShadowEffect* pEffect = new QGraphicsDropShadowEffect;
	pEffect->setColor(Qt::white);
	pEffect->setOffset(0);
	pEffect->setBlurRadius(2);
	setGraphicsEffect(pEffect);
}

void ZenoRampSelector::initRampPos(const QPointF& pos, const QColor& clr)
{
	setFlag(ItemSendsScenePositionChanges, false);
	setPos(pos);
	setFlag(ItemSendsScenePositionChanges, true);
	setBrush(clr);
}

void ZenoRampSelector::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	Q_UNUSED(widget);
	painter->setRenderHint(QPainter::Antialiasing, true);
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
			QPen pen(QColor(255, 128, 0), 1);
			setPen(pen);
		}
		else
		{
			QColor borderClr(0, 0, 0);
			QPen pen(borderClr, 1);
			setPen(pen);
		}
	}
	else if (change == QGraphicsItem::ItemPositionChange)
	{
		QPointF fixPos(value.toPointF().x(), pos().y());
		return fixPos;
	}
	else if (change == QGraphicsItem::ItemPositionHasChanged)
	{
		QPointF wtf = value.toPointF();
		QPointF pos = this->pos();
		m_rampBar->updateRampPos(this);
	}
	return _base::itemChange(change, value);
}


////////////////////////////////////////////////////////////////////
ZenoRampGroove::ZenoRampGroove(QGraphicsItem* parent)
	: _base(parent)
{
	QPen pen(QColor(45, 60, 76), 1);
	setPen(pen);

	QGraphicsDropShadowEffect* pEffect = new QGraphicsDropShadowEffect;
	pEffect->setColor(Qt::white);
	pEffect->setOffset(0);
	pEffect->setBlurRadius(3);

	setGraphicsEffect(pEffect);
}


/////////////////////////////////////////////////////////////////////
ZenoRampBar::ZenoRampBar(QWidget* parent)
	: QGraphicsView(parent)
	, m_barHeight(ZenoStyle::dpiScaled(32))
	, m_pColorItem(nullptr)
	, m_szSelector(ZenoStyle::dpiScaled(10))
{
	setFixedHeight(m_barHeight);

	m_scene = new QGraphicsScene;
	m_pColorItem = new QGraphicsRectItem(0, 0, 272, m_barHeight);

	ZenoRampGroove* pLineItem = new ZenoRampGroove;
	int y = m_barHeight / 2;
	pLineItem->setLine(0, y, 272, y);

	m_scene->addItem(m_pColorItem);
	m_scene->addItem(pLineItem);

	setScene(m_scene);
}

void ZenoRampBar::initRamps(const COLOR_RAMPS& ramps)
{
	QLinearGradient grad(0, 0, 272, 0);
	for (COLOR_RAMP ramp : ramps)
	{
		int y = m_barHeight / 2;
		ZenoRampSelector* selector = new ZenoRampSelector(this);
		selector->setRect(0, 0, m_szSelector, m_szSelector);
		qreal xPos = 272 * ramp.pos, yPos = (m_barHeight - m_szSelector) / 2.;

		QColor clr(ramp.r * 255, ramp.g * 255, ramp.b * 255);
		xPos = qMin(272. - 10, xPos);
		selector->initRampPos(QPointF(xPos, yPos), clr);
		m_scene->addItem(selector);
		grad.setColorAt(ramp.pos, clr);

		m_ramps[selector] = ramp;
	}
	m_pColorItem->setBrush(QBrush(grad));
}

void ZenoRampBar::removeRamp()
{
	for (ZenoRampSelector* pSelector : m_ramps.keys())
	{
		if (pSelector->isSelected())
		{
			m_ramps.remove(pSelector);
			m_scene->removeItem(pSelector);
		}
	}
	refreshBar();
}

void ZenoRampBar::newRamp()
{

}

COLOR_RAMPS ZenoRampBar::colorRamps() const
{
	auto vals = m_ramps.values();
	qSort(vals.begin(), vals.end(), [=](const COLOR_RAMP& lhs, const COLOR_RAMP& rhs) {
		return lhs.pos < rhs.pos;
		});
	return vals.toVector();
}

void ZenoRampBar::updateRampPos(ZenoRampSelector* pSelector)
{
	COLOR_RAMP& ramp = m_ramps[pSelector];
	ramp.pos = pSelector->x() / 272;
	refreshBar();
}

void ZenoRampBar::updateRampColor(const QColor& clr)
{
	for (ZenoRampSelector* pSelector : m_ramps.keys())
	{
		//find selection ramp.
		//todo: actually only one selector should be selected.
		if (pSelector->isSelected())
		{
			COLOR_RAMP& ramp = m_ramps[pSelector];
			ramp.r = clr.red() / 255.;
			ramp.g = clr.green() / 255.;
			ramp.b = clr.blue() / 255.;
		}
	}
	refreshBar();
}

void ZenoRampBar::refreshBar()
{
	QLinearGradient grad(0, 0, 272, 0);
	for (COLOR_RAMP rmp : m_ramps)
	{
		QColor clr(rmp.r * 255, rmp.g * 255, rmp.b * 255);
		grad.setColorAt(rmp.pos, clr);
	}
	m_pColorItem->setBrush(QBrush(grad));
}


/////////////////////////////////////////////////////////////////////
ZenoHeatMapEditor::ZenoHeatMapEditor(const COLOR_RAMPS& colorRamps, QWidget* parent)
	: QDialog(parent)
	, m_colorRamps(colorRamps)
{
	setWindowFlags(Qt::Dialog | Qt::MSWindowsFixedSizeDialogHint);
	m_ui = new Ui::HeatMapEditor;
	m_ui->setupUi(this);
	init();
	initSignals();
	installFilters();
}

ZenoHeatMapEditor::~ZenoHeatMapEditor()
{
}

void ZenoHeatMapEditor::init()
{
	initRamps();
	m_ui->cbPreset->addItems({"BlackBody", "Grayscale", "InfraRed", "TwoTone", "WhiteToRed"});
	m_ui->hueSlider;
}

COLOR_RAMPS ZenoHeatMapEditor::colorRamps() const
{
	return m_ui->rampBarView->colorRamps();
}

void ZenoHeatMapEditor::installFilters()
{
}

void ZenoHeatMapEditor::initSignals()
{
	connect(m_ui->btnAdd, SIGNAL(clicked()), this, SLOT(onAddRampBtnClicked()));
	connect(m_ui->btnDelete, SIGNAL(clicked()), this, SLOT(onRemoveRampBtnClicked()));
}

void ZenoHeatMapEditor::initRamps()
{
	m_ui->rampBarView->initRamps(m_colorRamps);
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

void ZenoHeatMapEditor::onAddRampBtnClicked()
{
	m_ui->rampBarView->newRamp();
}

void ZenoHeatMapEditor::onRemoveRampBtnClicked()
{
	m_ui->rampBarView->removeRamp();
}