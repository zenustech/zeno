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
	, m_currSelector(nullptr)
	, m_pLineItem(nullptr)
{
	setFixedHeight(m_barHeight);

	int width = this->width();

	m_scene = new QGraphicsScene;
	m_pColorItem = new QGraphicsRectItem;

	m_pLineItem = new ZenoRampGroove;
	int y = m_barHeight / 2;

	m_scene->addItem(m_pColorItem);
	m_scene->addItem(m_pLineItem);

	setScene(m_scene);
}

void ZenoRampBar::onSelectionChanged()
{
	bool noSelection = true;
	for (auto iter = m_ramps.begin(); iter != m_ramps.end(); iter++)
	{
		ZenoRampSelector* pSelector = iter.key();
		if (pSelector->isSelected() && m_currSelector != pSelector)
		{
			noSelection = false;
			if (m_currSelector != pSelector)
			{
				if (m_currSelector)
					m_currSelector->setSelected(false);
				m_currSelector = pSelector;
			}
			break;
		}
	}
	if (noSelection)
		m_currSelector->setSelected(true);
}

void ZenoRampBar::initRamps(int width)
{
	m_width = width;
	m_pColorItem->setRect(0, 0, m_width, m_barHeight);
	int y = m_barHeight / 2;
	m_pLineItem->setLine(0, y, m_width, y);

	QLinearGradient grad(0, 0, m_width, 0);
	for (COLOR_RAMP ramp : m_initRamps)
	{
		int y = m_barHeight / 2;
		ZenoRampSelector* selector = new ZenoRampSelector(this);
		selector->setRect(0, 0, m_szSelector, m_szSelector);
		qreal xPos = m_width * ramp.pos, yPos = (m_barHeight - m_szSelector) / 2.;

		QColor clr(ramp.r * 255, ramp.g * 255, ramp.b * 255);
		xPos = qMin(m_width - 10., xPos);
		selector->initRampPos(QPointF(xPos, yPos), clr);
		m_scene->addItem(selector);
		grad.setColorAt(ramp.pos, clr);

		m_ramps[selector] = ramp;
	}
	m_pColorItem->setBrush(QBrush(grad));
	m_currSelector = m_ramps.firstKey();

	m_currSelector->setSelected(true);
	connect(m_scene, SIGNAL(selectionChanged()), this, SLOT(onSelectionChanged()));
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

COLOR_RAMP ZenoRampBar::colorRamp() const
{
	return m_ramps[m_currSelector];
}

void ZenoRampBar::resizeEvent(QResizeEvent* event)
{
	QSize sz = event->size();
	QGraphicsView::resizeEvent(event);
	initRamps(sz.width());
}

void ZenoRampBar::setColorRamps(const COLOR_RAMPS& ramps)
{
	m_initRamps = ramps;
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
	ramp.pos = pSelector->x() / m_width;
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
	QLinearGradient grad(0, 0, m_width, 0);
	for (COLOR_RAMP rmp : m_ramps)
	{
		QColor clr(rmp.r * 255, rmp.g * 255, rmp.b * 255);
		grad.setColorAt(rmp.pos, clr);
	}
	m_pColorItem->setBrush(QBrush(grad));
}


HSVSelctor::HSVSelctor(QGraphicsItem* parent)
	: QGraphicsEllipseItem(parent)
{
	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);
}

void HSVSelctor::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	Q_UNUSED(widget);
	painter->setRenderHint(QPainter::Antialiasing, true);
	painter->setPen(this->pen());
	painter->setBrush(Qt::NoBrush);
	if ((this->spanAngle() != 0) && (qAbs((this->spanAngle()) % (360 * 16) == 0)))
		painter->drawEllipse(rect());
	else
		painter->drawPie(rect(), startAngle(), spanAngle());
}

QVariant HSVSelctor::itemChange(GraphicsItemChange change, const QVariant& value)
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
		return value;
	}
	else if (change == QGraphicsItem::ItemPositionHasChanged)
	{
		//
	}
	return _base::itemChange(change, value);
}


/////////////////////////////////////////////////////////////////////
ZenoHSVColorView::ZenoHSVColorView(QWidget* parent)
	: QGraphicsView(parent)
	, m_pColorItem(nullptr)
{
	m_scene = new QGraphicsScene;
	m_pColorItem = new QGraphicsRectItem;
	m_pColorItem->setZValue(-10);

	m_selector = new HSVSelctor;
	m_selector->setRect(0, 0, 10, 10);
	m_selector->setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemIsSelectable);
	m_selector->setZValue(10);
	m_selector->setPos(QPointF(50, 50));

	m_scene->addItem(m_pColorItem);
	m_scene->addItem(m_selector);
	setScene(m_scene);
}

void ZenoHSVColorView::resizeEvent(QResizeEvent* event)
{
	QSize sz = event->size();
	QGraphicsView::resizeEvent(event);
	initHSVColorView(sz);
}

void ZenoHSVColorView::initHSVColorView(const QSize& sz)
{
	m_pColorItem->setRect(0, 0, sz.width(), sz.height());

	QLinearGradient grad(0, 0, sz.width(), sz.height());

	QGradientStops stops;
	stops.append(QGradientStop(0, QColor(0, 0, 0, 0)));
	stops.append(QGradientStop(0.5, QColor(0, 255, 0)));
	stops.append(QGradientStop(1.0, QColor(0, 0, 0, 255)));

	grad.setStops(stops);

	grad.setSpread(QGradient::PadSpread);

	QBrush brush(grad);
	m_pColorItem->setBrush(brush);
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
	m_ui->rampBarView->setColorRamps(m_colorRamps);
}

void ZenoHeatMapEditor::initColorView()
{
	COLOR_RAMP ramp = m_ui->rampBarView->colorRamp();
	QColor clr(ramp.r * 255, ramp.g * 255, ramp.b * 255);
	QColor hsvClr = clr.toHsv();
	int h = 0, s = 0, v = 0;
	hsvClr.getHsv(&h, &s, &v);

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