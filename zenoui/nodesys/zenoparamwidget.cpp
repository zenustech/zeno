#include "zenoparamwidget.h"
#include "zenosocketitem.h"
#include "../render/common_id.h"
#include "../style/zenostyle.h"


ZenoParamWidget::ZenoParamWidget(QGraphicsItem* parent, Qt::WindowFlags wFlags)
    : QGraphicsProxyWidget(parent, wFlags)
{
}

ZenoParamWidget::~ZenoParamWidget()
{
}

int ZenoParamWidget::type() const
{
    return Type;
}

void ZenoParamWidget::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QGraphicsProxyWidget::paint(painter, option, widget);
}

void ZenoParamWidget::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsProxyWidget::mousePressEvent(event);
}

void ZenoParamWidget::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsProxyWidget::mouseReleaseEvent(event);
}

void ZenoParamWidget::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsProxyWidget::mouseDoubleClickEvent(event);
    emit doubleClicked();
}


///////////////////////////////////////////////////////////////////////////////
ZenoFrame::ZenoFrame(QWidget* parent, Qt::WindowFlags f)
    : QFrame(parent, f)
{
	setFrameShape(QFrame::VLine);
    QPalette pal = palette();
    pal.setBrush(QPalette::WindowText, QColor(86, 96, 143));
    setPalette(pal);
    setLineWidth(4);
}

ZenoFrame::~ZenoFrame()
{
}

QSize ZenoFrame::sizeHint() const
{
    QSize sz = QFrame::sizeHint();
    return sz;
    //return QSize(4, sz.height());
}

void ZenoFrame::paintEvent(QPaintEvent* e)
{
    QFrame::paintEvent(e);
}


///////////////////////////////////////////////////////////////////////////////
ZenoGvLineEdit::ZenoGvLineEdit(QWidget* parent)
    : QLineEdit(parent)
{
    setAutoFillBackground(false);
}

void ZenoGvLineEdit::paintEvent(QPaintEvent* e)
{
    QLineEdit::paintEvent(e);
}


////////////////////////////////////////////////////////////////////////////////
ZenoParamLineEdit::ZenoParamLineEdit(const QString &text, LineEditParam param, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    m_pLineEdit = new ZenoGvLineEdit;
    m_pLineEdit->setText(text);
    m_pLineEdit->setTextMargins(param.margins);
    m_pLineEdit->setPalette(param.palette);
    m_pLineEdit->setFont(param.font);
    m_pLineEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setWidget(m_pLineEdit);
    connect(m_pLineEdit, SIGNAL(editingFinished()), this, SIGNAL(editingFinished()));
}

QString ZenoParamLineEdit::text() const
{
    return m_pLineEdit->text();
}

void ZenoParamLineEdit::setText(const QString &text)
{
    m_pLineEdit->setText(text);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoParamLabel::ZenoParamLabel(const QString &text, const QFont &font, const QBrush &fill, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    m_label = new QLabel(text);

    QPalette palette;
    palette.setColor(QPalette::WindowText, fill.color());
    m_label->setFont(font);
    m_label->setPalette(palette);
    m_label->setAttribute(Qt::WA_TranslucentBackground);
    m_label->setAutoFillBackground(true);
    setWidget(m_label);
}

void ZenoParamLabel::setAlignment(Qt::Alignment alignment)
{
    m_label->setAlignment(alignment);
}


////////////////////////////////////////////////////////////////////////////////////
ZComboBoxItemDelegate::ZComboBoxItemDelegate(ComboBoxParam param, QObject *parent)
    : QStyledItemDelegate(parent)
    , m_param(param)
{
}

void ZComboBoxItemDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);

    option->backgroundBrush.setStyle(Qt::SolidPattern);
    if (option->state & QStyle::State_MouseOver)
    {
        option->backgroundBrush.setColor(m_param.itemBgHovered);
    }
    else
    {
        option->backgroundBrush.setColor(m_param.itemBgNormal);
    }
}

void ZComboBoxItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);
    painter->fillRect(opt.rect, opt.backgroundBrush);
    painter->setPen(QPen(m_param.textColor));
    painter->drawText(opt.rect.adjusted(m_param.margins.left(), 0, 0, 0), opt.text);
}

QSize ZComboBoxItemDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    int w = ((QWidget *) parent())->width();
    return QSize(w, 24);
}


ZenoGvComboBox::ZenoGvComboBox(QWidget *parent)
    : QComboBox(parent)
{
}

void ZenoGvComboBox::paintEvent(QPaintEvent *e)
{
    QComboBox::paintEvent(e);
}

ZenoParamComboBox::ZenoParamComboBox(const QStringList &items, ComboBoxParam param, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    m_combobox = new ZComboBox;
    m_combobox->addItems(items);
    m_combobox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_combobox->setItemDelegate(new ZComboBoxItemDelegate(param, m_combobox));
    setWidget(m_combobox);
    connect(m_combobox, SIGNAL(textActivated(const QString&)), this, SIGNAL(textActivated(const QString&)));
}

void ZenoParamComboBox::setText(const QString& text)
{
    m_combobox->setCurrentText(text);
}


////////////////////////////////////////////////////////////////////////////////////
ZenoParamPushButton::ZenoParamPushButton(const QString &name, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    QPushButton* pBtn = new QPushButton(name);
    //temp code:
    pBtn->setFixedWidth(20);
    QPalette palette;
    palette.setColor(QPalette::Base, QColor(50, 50, 50));
    palette.setColor(QPalette::Window, QColor(50, 50, 50));
    palette.setColor(QPalette::Text, Qt::white);
    pBtn->setPalette(palette);
    setWidget(pBtn);
}


////////////////////////////////////////////////////////////////////////////////////
ZenoParamOpenPath::ZenoParamOpenPath(const QString &filename, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_path(filename)
{
    QLineEdit *plineEdit = new QLineEdit(filename);
    QPushButton *openBtn = new QPushButton("...");
    QWidget* pWidget = new QWidget;
    QHBoxLayout *pLayout = new QHBoxLayout;
    plineEdit->setReadOnly(true);
    pLayout->addWidget(plineEdit);
    pLayout->addWidget(openBtn);
    pLayout->setMargin(0);
    pWidget->setLayout(pLayout);
    pWidget->setAutoFillBackground(true);
    setWidget(pWidget);
}


//////////////////////////////////////////////////////////////////////////////////////
ZenoParamMultilineStr::ZenoParamMultilineStr(const QString &value, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_value(value)
    , m_pTextEdit(nullptr)
{
    m_pTextEdit = new QTextEdit;
    m_pTextEdit->setText(value);
    setWidget(m_pTextEdit);
    connect(m_pTextEdit, SIGNAL(textChanged()), this, SIGNAL(textChanged()));
    m_pTextEdit->installEventFilter(this);
}

void ZenoParamMultilineStr::setText(const QString& text)
{
    m_pTextEdit->setText(text);
}

QString ZenoParamMultilineStr::text() const
{
    return m_pTextEdit->toPlainText();
}

bool ZenoParamMultilineStr::eventFilter(QObject* object, QEvent* event)
{
    if (object == m_pTextEdit && event->type() == QEvent::FocusOut)
    {
        emit editingFinished();
    }
    return ZenoParamWidget::eventFilter(object, event);
}


//////////////////////////////////////////////////////////////////////////////////////
ZenoTextLayoutItem::ZenoTextLayoutItem(const QString &text, const QFont &font, const QColor &color, QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , QGraphicsTextItem(text, parent)
    , m_text(text)
{
    setZValue(ZVALUE_ELEMENT);
    setFont(font);
    setDefaultTextColor(color);
    
    setGraphicsItem(this);
    setFlags(ItemSendsScenePositionChanges);
}

void ZenoTextLayoutItem::setGeometry(const QRectF& geom)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(geom);
    setPos(geom.topLeft());
    //emit geometrySetup(scenePos());
}

QRectF ZenoTextLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoTextLayoutItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    //painter->setBrush(Qt::black);
    //painter->drawRect(boundingRect());  //FOR DEBUG LAOUT POSITION
    QGraphicsTextItem::paint(painter, option, widget);
}

QSizeF ZenoTextLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    QRectF rc = QGraphicsTextItem::boundingRect();
    switch (which)
    {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
            return rc.size();
        case Qt::MaximumSize:
            return QSizeF(1000, rc.height());
        default:
            break;
    }
    return constraint;
}


//////////////////////////////////////////////////////////////////////////////////////
ZenoSpacerItem::ZenoSpacerItem(bool bHorizontal, qreal size, QGraphicsItem* parent)
    : QGraphicsLayoutItem()
    , QGraphicsItem(parent)
    , m_bHorizontal(bHorizontal)
    , m_size(size)
{
}

void ZenoSpacerItem::setGeometry(const QRectF& rect)
{
	prepareGeometryChange();
	QGraphicsLayoutItem::setGeometry(rect);
	setPos(rect.topLeft());
}

QRectF ZenoSpacerItem::boundingRect() const
{
    if (m_bHorizontal)
        return QRectF(0, 0, m_size, 0);
    else
        return QRectF(0, 0, 0, m_size);
}

void ZenoSpacerItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}

QSizeF ZenoSpacerItem::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
	QRectF rc = boundingRect();
	switch (which)
	{
	case Qt::MinimumSize:
	case Qt::PreferredSize:
    case Qt::MaximumSize:
		return rc.size();
	default:
		break;
	}
	return constraint;
}

//////////////////////////////////////////////////////////////////////////////////////
ZenoBoardTextLayoutItem::ZenoBoardTextLayoutItem(const QString &text, const QFont &font, const QColor &color, const QSizeF& sz, QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , QGraphicsTextItem(text, parent)
    , m_text(text)
    , m_size(sz)
{
    setZValue(ZVALUE_ELEMENT);
    setFont(font);
    setDefaultTextColor(color);

    setGraphicsItem(this);
    setFlags(ItemIsFocusable | ItemIsSelectable | ItemSendsScenePositionChanges);
    setTextInteractionFlags(Qt::TextEditorInteraction);

    connect(document(), &QTextDocument::contentsChanged, this, [=]() {
        updateGeometry();
    });
}

void ZenoBoardTextLayoutItem::setGeometry(const QRectF& geom)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(geom);
    setPos(geom.topLeft());
    //emit geometrySetup(scenePos());
}

QRectF ZenoBoardTextLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoBoardTextLayoutItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QGraphicsTextItem::paint(painter, option, widget);
}

QSizeF ZenoBoardTextLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    return m_size;
    QRectF rc = QGraphicsTextItem::boundingRect();
    switch (which) {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
            return rc.size();
        case Qt::MaximumSize:
            return QSizeF(1000, 1000);
        default:
            break;
    }
    return constraint;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoMinStatusBtnItem::ZenoMinStatusBtnItem(const StatusComponent& statusComp, QGraphicsItem* parent)
    : _base(parent)
    , m_minMute(nullptr)
    , m_minView(nullptr)
    , m_minOnce(nullptr)
{
    m_minMute = new ZenoImageItem(statusComp.mute, QSizeF(33, 42), this);
    m_minView = new ZenoImageItem(statusComp.view, QSizeF(25, 42), this);
    m_minOnce = new ZenoImageItem(statusComp.once, QSizeF(33, 42), this);
	m_once = new ZenoImageItem(":/icons/ONCE_dark.svg", ":/icons/ONCE_light.svg", ":/icons/ONCE_light.svg", QSize(50, 42), this);
	m_mute = new ZenoImageItem(":/icons/MUTE_dark.svg", ":/icons/MUTE_light.svg", ":/icons/MUTE_light.svg", QSize(50, 42), this);
	m_view = new ZenoImageItem(":/icons/VIEW_dark.svg", ":/icons/VIEW_light.svg", ":/icons/VIEW_light.svg", QSize(50, 42), this);
    m_minMute->setCheckable(true);
    m_minView->setCheckable(true);
    m_minOnce->setCheckable(true);
    m_once->setCheckable(true);
    m_mute->setCheckable(true);
    m_view->setCheckable(true);
    m_once->hide();
    m_mute->hide();
    m_view->hide();

    m_minOnce->setPos(QPointF(0, 0));
    m_minMute->setPos(QPointF(20, 0));
    m_minView->setPos(QPointF(40, 0));

    QSizeF sz2 = m_once->size();
	QPointF base = QPointF(12, -sz2.height() - 8);
	m_once->setPos(base);
	base += QPointF(38, 0);
	m_mute->setPos(base);
	base += QPointF(38, 0);
	m_view->setPos(base);

    m_minOnce->setZValue(ZVALUE_ELEMENT);
    m_minView->setZValue(ZVALUE_ELEMENT);
    m_minMute->setZValue(ZVALUE_ELEMENT);

    connect(m_minOnce, SIGNAL(hoverChanged(bool)), m_once, SLOT(setHovered(bool)));
    connect(m_minView, SIGNAL(hoverChanged(bool)), m_view, SLOT(setHovered(bool)));
    connect(m_minMute, SIGNAL(hoverChanged(bool)), m_mute, SLOT(setHovered(bool)));
	connect(m_once, SIGNAL(hoverChanged(bool)), m_minOnce, SLOT(setHovered(bool)));
	connect(m_view, SIGNAL(hoverChanged(bool)), m_minView, SLOT(setHovered(bool)));
	connect(m_mute, SIGNAL(hoverChanged(bool)), m_minMute, SLOT(setHovered(bool)));

	connect(m_minOnce, SIGNAL(toggled(bool)), m_once, SLOT(toggle(bool)));
	connect(m_minView, SIGNAL(toggled(bool)), m_view, SLOT(toggle(bool)));
	connect(m_minMute, SIGNAL(toggled(bool)), m_mute, SLOT(toggle(bool)));
	connect(m_once, SIGNAL(toggled(bool)), m_minOnce, SLOT(toggle(bool)));
	connect(m_view, SIGNAL(toggled(bool)), m_minView, SLOT(toggle(bool)));
	connect(m_mute, SIGNAL(toggled(bool)), m_minMute, SLOT(toggle(bool)));

    connect(m_minMute, &ZenoImageItem::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_MUTE, hovered);
    });
	connect(m_minView, &ZenoImageItem::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_VIEW, hovered);
    });
	connect(m_minOnce, &ZenoImageItem::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_ONCE, hovered);
	});

    setAcceptHoverEvents(true);
}

void ZenoMinStatusBtnItem::setOptions(int options)
{
    if (options & OPT_ONCE)
    {
        setChecked(STATUS_ONCE, true);
    }
    else if (options & OPT_MUTE)
    {
        setChecked(STATUS_MUTE, true);
    }
    else if (options & OPT_VIEW)
    {
        setChecked(STATUS_VIEW, true);
    }
}

void ZenoMinStatusBtnItem::setChecked(STATUS_BTN btn, bool bChecked)
{
    if (btn == STATUS_MUTE)
    {
        m_mute->toggle(bChecked);
        m_minMute->toggle(bChecked);
    }
    if (btn == STATUS_ONCE)
    {
		m_once->toggle(bChecked);
		m_minOnce->toggle(bChecked);
    }
	if (btn == STATUS_VIEW)
	{
		m_view->toggle(bChecked);
		m_minView->toggle(bChecked);
	}
}

void ZenoMinStatusBtnItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    m_mute->show();
    m_view->show();
    m_once->show();
    _base::hoverEnterEvent(event);
}

void ZenoMinStatusBtnItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void ZenoMinStatusBtnItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
	m_mute->hide();
	m_view->hide();
	m_once->hide();
    _base::hoverLeaveEvent(event);
}

QRectF ZenoMinStatusBtnItem::boundingRect() const
{
    if (!m_mute->isVisible() && !m_view->isVisible() && !m_once->isVisible())
    {
        QRectF rc;
		rc = m_minMute->sceneBoundingRect();
		rc |= m_minView->sceneBoundingRect();
		rc |= m_minOnce->sceneBoundingRect();
        rc = mapRectFromScene(rc);
        return rc;
    }
    else
    {
		QRectF rc = childrenBoundingRect();
		return rc;
    }
}

void ZenoMinStatusBtnItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoMinStatusBtnWidget::ZenoMinStatusBtnWidget(const StatusComponent& statusComp, QGraphicsItem* parent)
	: QGraphicsLayoutItem()
	, ZenoMinStatusBtnItem(statusComp, parent)
{
}

void ZenoMinStatusBtnWidget::updateGeometry()
{
    QGraphicsLayoutItem::updateGeometry();
}

void ZenoMinStatusBtnWidget::setGeometry(const QRectF& rect)
{
	prepareGeometryChange();
	QGraphicsLayoutItem::setGeometry(rect);
	setPos(rect.topLeft());
}

QRectF ZenoMinStatusBtnWidget::boundingRect() const
{
    return ZenoMinStatusBtnItem::boundingRect();
	QRectF rc = QRectF(QPointF(0, 0), geometry().size());
	return rc;
}

QSizeF ZenoMinStatusBtnWidget::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
	switch (which)
	{
	case Qt::MinimumSize:
	case Qt::PreferredSize:
	case Qt::MaximumSize:
    {
        QRectF rc = m_minMute->sceneBoundingRect();
        rc |= m_minOnce->sceneBoundingRect();
        rc |= m_minView->sceneBoundingRect();
        return rc.size();
    }
	default:
		break;
	}
	return constraint;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoSvgLayoutItem::ZenoSvgLayoutItem(const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , ZenoImageItem(elem, sz, parent)
{
    setGraphicsItem(this);
}

QSizeF ZenoSvgLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    switch (which)
    {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
        case Qt::MaximumSize:
            return ZenoImageItem::boundingRect().size();
        default:
            break;
    }
    return constraint;
}

QRectF ZenoSvgLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoSvgLayoutItem::setGeometry(const QRectF &rect)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(rect);
    setPos(rect.topLeft());
}

void ZenoSvgLayoutItem::updateGeometry()
{
    QGraphicsLayoutItem::updateGeometry();
}


/////////////////////////////////////////////////////////////////////////
SpacerLayoutItem::SpacerLayoutItem(QSizeF sz, bool bHorizontal, QGraphicsLayoutItem *parent, bool isLayout)
    : QGraphicsLayoutItem(parent, isLayout)
    , m_sz(sz)
{
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
}

QSizeF SpacerLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    switch (which)
    {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
            return m_sz;
        case Qt::MaximumSize:
            return m_sz;
            QSizeF(1000, 1000);
        default:
            return m_sz;
    }
    return constraint;
}