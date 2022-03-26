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
	, m_pColorItem(nullptr)
	, m_szSelector(ZenoStyle::dpiScaled(10))
	, m_currSelector(nullptr)
	, m_pLineItem(nullptr)
{
	m_scene = new QGraphicsScene;

	setScene(m_scene);
	connect(m_scene, SIGNAL(selectionChanged()), this, SLOT(onSelectionChanged()));
}

void ZenoRampBar::onSelectionChanged()
{
	if (m_ramps.empty())
		return;

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

	Q_ASSERT(m_ramps.find(m_currSelector) != m_ramps.end());
	emit rampSelected(m_ramps[m_currSelector]);
}

void ZenoRampBar::onResizeInit(QSize sz)
{
	int W = sz.width();
	int H = sz.height();
	m_pColorItem->setRect(0, 0, W, H);
	m_pLineItem->setLine(0, H / 2, W, H / 2);
	m_grad.setFinalStop(W, 0);

	for (auto iter = m_ramps.begin(); iter != m_ramps.end(); iter++)
	{
		ZenoRampSelector* pSelector = iter.key();
		QGradientStop& ramp = iter.value();
		QColor clr = ramp.second;

		qreal xPos = W * ramp.first, yPos = (H - m_szSelector) / 2.;
		xPos = qMin(W - 10., xPos);
		pSelector->initRampPos(QPointF(xPos, yPos), clr);
	}
	refreshBar();
}

void ZenoRampBar::setColorRamps(const QLinearGradient& grad)
{
	m_grad = grad;
	m_grad.setFinalStop(width(), 0);

	m_currSelector = nullptr;
	m_ramps.clear();
	m_scene->clear();

	m_pColorItem = new QGraphicsRectItem;
	m_pLineItem = new ZenoRampGroove;
	m_scene->addItem(m_pColorItem);
	m_scene->addItem(m_pLineItem);

	int W = width();
	int H = height();

	m_pColorItem->setRect(0, 0, W, H);
	int y = H / 2;
	m_pLineItem->setLine(0, y, W, y);

	for (QGradientStop stop : grad.stops())
	{
		ZenoRampSelector* selector = new ZenoRampSelector(this);
		selector->setRect(0, 0, m_szSelector, m_szSelector);
		qreal pos = stop.first;
		QColor clr = stop.second;
        qreal xPos = W * pos, yPos = (H - m_szSelector) / 2.;

		xPos = qMin(W - 10., xPos);
		selector->initRampPos(QPointF(xPos, yPos), clr);

		m_scene->addItem(selector);

		m_ramps[selector] = stop;
	}

	BlockSignalScope scope(m_scene);
	onSelectionChanged();

	refreshBar();
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
	typedef std::pair<ZenoRampSelector*, QGradientStop> _PAIR;

	std::vector<_PAIR> L;
	for (ZenoRampSelector* pSelector : m_ramps.keys())
	{
		QGradientStop ramp = m_ramps[pSelector];
		L.push_back(std::pair(pSelector, ramp));
	}

	std::sort(L.begin(), L.end(), [=](const _PAIR& lhs, const _PAIR& rhs) {
		return lhs.first < rhs.first;
	});

	for (int i = 0; i < L.size(); i++)
	{
		if (m_currSelector == L[i].first && i + 1 < L.size())
		{
			QGradientStop p = L[i].second;
			QGradientStop n = L[i + 1].second;

			QGradientStop stop;
			stop.first = (p.first + n.first) / 2;
			stop.second = (p.second.rgb() + n.second.rgb()) / 2;

			ZenoRampSelector* selector = new ZenoRampSelector(this);
			m_ramps[selector] = stop;

			qreal xPos = width() * stop.first, yPos = (height() - m_szSelector) / 2.;
			selector->initRampPos(QPointF(xPos, yPos), stop.second);

			m_scene->addItem(selector);

			break;
		}
	}
	refreshBar();
}

QGradientStop ZenoRampBar::colorRamp() const
{
	Q_ASSERT(m_ramps.find(m_currSelector) != m_ramps.end());
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
	onResizeInit(sz);
}

QLinearGradient ZenoRampBar::colorRamps() const
{
	return m_grad;
}

void ZenoRampBar::updateRampPos(ZenoRampSelector* pSelector)
{
	Q_ASSERT(m_ramps.find(pSelector) != m_ramps.end());
	QGradientStop& ramp = m_ramps[pSelector];
	m_grad.stops();
	ramp.first = pSelector->x() / width();
	refreshBar();
}

void ZenoRampBar::updateRampColor(const QColor& clr)
{
	Q_ASSERT(m_ramps.find(m_currSelector) != m_ramps.end());
	QGradientStop& ramp = m_ramps[m_currSelector];
	if (clr == ramp.second)
		return;

	ramp.second = clr;
	m_grad.setColorAt(ramp.first, ramp.second);
	m_currSelector->setBrush(clr);
	refreshBar();
}

void ZenoRampBar::refreshBar()
{
	QGradientStops stops;
	for (QGradientStop stop : m_ramps)
	{
		stops.append(stop);
	}
	m_grad.setStops(stops);
	m_pColorItem->setBrush(QBrush(m_grad));
}


SVColorView::SVColorView(QWidget* parent)
	: QWidget(parent)
{
}

QColor SVColorView::color() const
{
	return m_color;
}

void SVColorView::setColor(const QColor& clr)
{
	m_color = clr;
	update();
	emit colorChanged(m_color);
}

void SVColorView::mousePressEvent(QMouseEvent* event)
{
	QWidget::mousePressEvent(event);
	updateColorByMouse(event->pos());
}

void SVColorView::mouseMoveEvent(QMouseEvent* event)
{
	QWidget::mouseMoveEvent(event);
	updateColorByMouse(event->pos());
}

void SVColorView::updateColorByMouse(const QPointF& pos)
{
	qreal m_H = m_color.hueF();
	qreal m_S = pos.x() / this->width();
	qreal m_V = 1 - pos.y() / this->height();
	setColor(QColor::fromHsvF(m_H, m_S, m_V));
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
	qreal h = 0, s = 0, v = 0;
	m_color.getHsvF(&h, &s, &v);
	QPointF center(s * this->width(), (1 - v) * this->height());
	painter.setPen(QPen(QColor(0,0,0)));
	painter.drawEllipse(center, nLenSelector, nLenSelector);
}


/////////////////////////////////////////////////////////////////////
ZenoHeatMapEditor::ZenoHeatMapEditor(const QLinearGradient& grad, QWidget* parent)
	: QDialog(parent)
{
	setWindowFlags(Qt::Dialog | Qt::MSWindowsFixedSizeDialogHint);
	m_ui = new Ui::HeatMapEditor;
	m_ui->setupUi(this);
	init(grad);
	initSignals();
	installFilters();
}

ZenoHeatMapEditor::~ZenoHeatMapEditor()
{
}

void ZenoHeatMapEditor::init(const QLinearGradient& grad)
{
	initRamps(grad);
	m_ui->cbPreset->addItems({"BlackBody", "Grayscale", "InfraRed", "TwoTone", "WhiteToRed"});
	m_ui->hueSlider;
	//m_ui->clrHex->setFont(QFont("HarmonyOS Sans", 10));
	initColorView();
}

QLinearGradient ZenoHeatMapEditor::colorRamps() const
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
	connect(m_ui->rampBarView, SIGNAL(rampSelected(QGradientStop)), this, SLOT(onRampColorClicked(QGradientStop)));
	connect(m_ui->hvColorView, SIGNAL(colorChanged(const QColor&)), this, SLOT(setColor(const QColor&)));
	connect(m_ui->spRedOrH, SIGNAL(valueChanged(int)), this, SLOT(onRedChanged(int)));
	connect(m_ui->spGreenOrS, SIGNAL(valueChanged(int)), this, SLOT(onGreenChanged(int)));
	connect(m_ui->spBlueOrV, SIGNAL(valueChanged(int)), this, SLOT(onBlueChanged(int)));
	connect(m_ui->hueSlider, SIGNAL(valueChanged(int)), this, SLOT(onHueChanged(int)));
	connect(m_ui->cbPreset, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(onCurrentIndexChanged(const QString&)));
	connect(m_ui->clrHex, SIGNAL(editingFinished()), this, SLOT(onClrHexEditFinished()));
}

void ZenoHeatMapEditor::onClrHexEditFinished()
{
	const QString& text = m_ui->clrHex->text();
	QColor clr(text);
	setColor(clr);
}

void ZenoHeatMapEditor::onCurrentIndexChanged(const QString& text)
{
	COLOR_RAMPS ramps;
	QLinearGradient grad;
	if (text == "BlackBody")
	{
		QGradientStops stops;
		stops.append({ 0., QColor::fromRgbF(0, 0, 0) });
		stops.append({ 0.33, QColor::fromRgbF(1, 0, 0) });
		stops.append({ 0.66, QColor::fromRgbF(1, 1, 0) });
		stops.append({ 1, QColor::fromRgbF(1, 1, 1) });
		grad.setStops(stops);
	}
	else if (text == "Grayscale")
	{
		QGradientStops stops;
		stops.append({ 0., QColor::fromRgbF(0, 0, 0) });
		stops.append({ 1, QColor::fromRgbF(1, 1, 1) });
		grad.setStops(stops);
	}
	else if (text == "InfraRed")
	{
		QGradientStops stops;
		stops.append({ 0., QColor::fromRgbF(0.2, 0, 1) });
		stops.append({ 0.25, QColor::fromRgbF(0, 0.85, 1) });
		stops.append({ 0.5, QColor::fromRgbF(0, 1, 0.1) });
		stops.append({ 0.75, QColor::fromRgbF(0.95, 1, 0) });
		stops.append({ 1, QColor::fromRgbF(1, 0, 0) });
		grad.setStops(stops);
	}
	else if (text == "TwoTone")
	{
		QGradientStops stops;
		stops.append({ 0., QColor::fromRgbF(0, 1, 1) });
		stops.append({ 0.49, QColor::fromRgbF(0, 0, 1) });
		stops.append({ 0.5, QColor::fromRgbF(1, 1, 1) });
		stops.append({ 0.51, QColor::fromRgbF(1, 0, 0) });
		stops.append({ 1, QColor::fromRgbF(1, 1, 0) });
		grad.setStops(stops);
	}
	else if (text == "WhiteToRed")
	{
		QGradientStops stops;
		stops.append({ 0., QColor::fromRgbF(1, 1, 1) });
		stops.append({ 1, QColor::fromRgbF(1, 0, 0) });
		grad.setStops(stops);
	}
	m_ui->rampBarView->setColorRamps(grad);
}

void ZenoHeatMapEditor::onRedChanged(int value)
{
	QColor clr = m_ui->hvColorView->color();
	clr.setRed(value);
	if (clr != m_ui->hvColorView->color())
		setColor(clr);
}

void ZenoHeatMapEditor::onGreenChanged(int value)
{
	QColor clr = m_ui->hvColorView->color();
	clr.setGreen(value);
	if (clr != m_ui->hvColorView->color())
		setColor(clr);
}

void ZenoHeatMapEditor::onBlueChanged(int value)
{
	QColor clr = m_ui->hvColorView->color();
	clr.setBlue(value);
	if (clr != m_ui->hvColorView->color())
		setColor(clr);
}

void ZenoHeatMapEditor::onHueChanged(int value)
{
	int hueValue = 360 - value;
	QColor clr = m_ui->hvColorView->color();
	int s = clr.saturation();
	int v = clr.value();
	clr.setHsv(hueValue, s, v);
	if (clr != m_ui->hvColorView->color())
		setColor(clr);
}

void ZenoHeatMapEditor::initRamps(const QLinearGradient& grad)
{
	m_ui->rampBarView->setColorRamps(grad);
}

void ZenoHeatMapEditor::initColorView()
{
	QGradientStop ramp = m_ui->rampBarView->colorRamp();
	m_ui->hvColorView->setColor(ramp.second);
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

void ZenoHeatMapEditor::onRampColorClicked(QGradientStop ramp)
{
	setColor(ramp.second);
}

void ZenoHeatMapEditor::setColor(const QColor& clr)
{
	BlockSignalScope scope1(m_ui->hvColorView);
	BlockSignalScope scope2(m_ui->hueSlider);
	BlockSignalScope scope3(m_ui->spRedOrH);
	BlockSignalScope scope4(m_ui->spGreenOrS);
	BlockSignalScope scope5(m_ui->spBlueOrV);
	BlockSignalScope scope6(m_ui->clrHex);
	BlockSignalScope scope7(m_ui->rampBarView);

	m_ui->hvColorView->setColor(clr);
	int hueSliderValue = 360 - clr.hue();
	m_ui->hueSlider->setValue(hueSliderValue);
	m_ui->spRedOrH->setValue(clr.red());
	m_ui->spGreenOrS->setValue(clr.green());
	m_ui->spBlueOrV->setValue(clr.blue());
	m_ui->clrHex->setText(clr.name());
	m_ui->rampBarView->updateRampColor(clr);
}