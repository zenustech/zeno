#include "ui_zenoheatmapeditor.h"
#include "zenoheatmapeditor.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/nodesys/zenosvgitem.h>
#include <zenoui/util/uihelper.h>


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
	if (change == QGraphicsItem::ItemSelectedChange)
	{
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

	connect(m_scene, SIGNAL(selectionChanged()), this, SLOT(onSelectionChanged()));
}

void ZenoRampBar::onSelectionChanged()
{
	BlockSignalScope scope(m_scene);
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
	if (m_currSelector == nullptr)
	{
		m_currSelector = m_ramps.firstKey();
		Q_ASSERT(m_currSelector);
	}
	if (noSelection)
		m_currSelector->setSelected(true);
	m_currSelector->update();

	emit rampSelected(m_ramps[m_currSelector]);
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
	BlockSignalScope scope(m_scene);
	onSelectionChanged();
}

void ZenoRampBar::removeRamp()
{
	for (ZenoRampSelector* pSelector : m_ramps.keys())
	{
		if (pSelector->isSelected())
		{
			if (m_ramps.size() > 2)
			{
				BlockSignalScope scope(m_scene);
				m_ramps.remove(pSelector);
				m_scene->removeItem(pSelector);
				m_currSelector = nullptr;
				onSelectionChanged();
				break;
			}
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

void ZenoRampBar::mousePressEvent(QMouseEvent* event)
{
	QGraphicsView::mousePressEvent(event);
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


SVColorView::SVColorView(QWidget* parent)
	: QWidget(parent)
{
}

void SVColorView::setColor(const QColor& clr)
{
	m_color = clr;
	update();
}

void SVColorView::mousePressEvent(QMouseEvent* event)
{
	QWidget::mousePressEvent(event);

	QPointF pos = event->pos();
	qreal m_H = m_color.hueF();
	qreal m_S = pos.x() / this->width();
	qreal m_V = pos.y() / this->height();

	m_color.setHsvF(m_H, m_S, m_V);
	emit colorChanged(m_color);
	update();
}

void SVColorView::mouseMoveEvent(QMouseEvent* event)
{
	QWidget::mouseMoveEvent(event);

	QPointF pos = event->pos();
	qreal m_H = m_color.hueF();
	qreal m_S = pos.x() / this->width();
	qreal m_V = pos.y() / this->height();

	m_color.setHsvF(m_H, m_S, m_V);
	emit colorChanged(m_color);
	update();
}

void SVColorView::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);

	painter.setRenderHint(QPainter::Antialiasing);

	QRect rect = this->rect();

	QLinearGradient linearGradientH(rect.topLeft(), rect.topRight());
	linearGradientH.setColorAt(0, QColor(255, 255, 255));
	QColor color;
	color.setHsvF(m_color.hueF(), 1, 1);
	linearGradientH.setColorAt(1, color);
	painter.fillRect(rect, linearGradientH);

	QLinearGradient linearGradientV(rect.topLeft(), rect.bottomLeft());
	linearGradientV.setColorAt(0, QColor(0, 0, 0, 0));
	linearGradientV.setColorAt(1, QColor(0, 0, 0, 255));
	painter.fillRect(rect, linearGradientV);

	static const int nLenSelector = ZenoStyle::dpiScaled(6);
	QPointF center(m_color.saturationF() * this->width(), m_color.valueF() * this->height());
	painter.setPen(QPen(QColor(0,0,0)));
	painter.drawEllipse(center, nLenSelector, nLenSelector);
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
	initColorView();
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
	connect(m_ui->rampBarView, SIGNAL(rampSelected(COLOR_RAMP)), this, SLOT(onRampColorClicked(COLOR_RAMP)));
}

void ZenoHeatMapEditor::initRamps()
{
	m_ui->rampBarView->setColorRamps(m_colorRamps);
}

void ZenoHeatMapEditor::initColorView()
{
	COLOR_RAMP ramp = m_ui->rampBarView->colorRamp();
	QColor clr(ramp.r * 255, ramp.g * 255, ramp.b * 255);
	m_ui->hvColorView->setColor(clr);
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

void ZenoHeatMapEditor::onRampColorClicked(COLOR_RAMP ramp)
{
	QColor clr(ramp.r, ramp.g, ramp.b);
	m_ui->hvColorView->setColor(clr);
	int h = clr.hue();
	m_ui->hueSlider->setValue(h);
}