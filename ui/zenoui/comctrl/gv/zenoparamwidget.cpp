#include "zenoparamwidget.h"
#include "zenosocketitem.h"
#include <zenoui/render/common_id.h>
#include <zenoui/style/zenostyle.h>


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
ZenoParamLineEdit::ZenoParamLineEdit(const QString &text, PARAM_CONTROL ctrl, LineEditParam param, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    m_pLineEdit = new QLineEdit;
    m_pLineEdit->setText(text);
    m_pLineEdit->setTextMargins(param.margins);
    m_pLineEdit->setPalette(param.palette);
    m_pLineEdit->setFont(param.font);
    m_pLineEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    switch (ctrl)
    {
    case CONTROL_INT:
        m_pLineEdit->setValidator(new QIntValidator);
        break;
    case CONTROL_FLOAT:
        m_pLineEdit->setValidator(new QDoubleValidator);
        break;
    case CONTROL_BOOL:
        break;
    }

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


///////////////////////////////////////////////////////////////////////////
ZenoVecEditWidget::ZenoVecEditWidget(const QVector<qreal>& vec, QGraphicsItem* parent)
    : ZenoParamWidget(parent)
    , m_pEdit(nullptr)
{
    m_pEdit = new ZVecEditor(vec, true, 3, "zenonode");
	setWidget(m_pEdit);
	connect(m_pEdit, SIGNAL(editingFinished()), this, SIGNAL(editingFinished()));
}

QVector<qreal> ZenoVecEditWidget::vec() const
{
    return m_pEdit->vec();
}

void ZenoVecEditWidget::setVec(const QVector<qreal>& vec)
{
    m_pEdit->onValueChanged(vec);
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

void ZenoParamLabel::setText(const QString& text)
{
    m_label->setText(text);
}

void ZenoParamLabel::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget*  widget)
{
    painter->fillRect(boundingRect(), QColor(0,0,0));
    ZenoParamWidget::paint(painter, option, widget);
}


////////////////////////////////////////////////////////////////////////////////////
ZComboBoxItemDelegate::ZComboBoxItemDelegate(QObject *parent)
    : QStyledItemDelegate(parent)
{
}

void ZComboBoxItemDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);

    option->backgroundBrush.setStyle(Qt::SolidPattern);
    if (option->state & QStyle::State_MouseOver)
    {
        option->backgroundBrush.setColor(QColor(23, 160, 252));
    }
    else
    {
        option->backgroundBrush.setColor(QColor(58, 58, 58));
    }
}

void ZComboBoxItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);
    painter->fillRect(opt.rect, opt.backgroundBrush);
    painter->setPen(QPen(QColor(210, 203, 197)));
    painter->drawText(opt.rect.adjusted(8, 0, 0, 0), opt.text);
}

QSize ZComboBoxItemDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    int w = ((QWidget *) parent())->width();
    return ZenoStyle::dpiScaledSize(QSize(w, 28));
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
    m_combobox->setItemDelegate(new ZComboBoxItemDelegate(m_combobox));
    setWidget(m_combobox);

    setZValue(ZVALUE_POPUPWIDGET);
    connect(m_combobox, SIGNAL(activated(int)), this, SLOT(onComboItemActivated(int)));
}

void ZenoParamComboBox::setText(const QString& text)
{
    m_combobox->setCurrentText(text);
}

QString ZenoParamComboBox::text()
{
    return m_combobox->currentText();
}

void ZenoParamComboBox::onComboItemActivated(int index)
{
    // pay attention to the compatiblity of qt!!!
    QString text = m_combobox->itemText(index);
    emit textActivated(text);
}


////////////////////////////////////////////////////////////////////////////////////
ZenoParamPushButton::ZenoParamPushButton(const QString &name, int width, QSizePolicy::Policy hor, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_width(width)
{
    QPushButton* pBtn = new QPushButton(name);
    pBtn->setProperty("cssClass", "grayButton");
    if (hor == QSizePolicy::Fixed)
        pBtn->setFixedWidth(width);
    pBtn->setSizePolicy(hor, QSizePolicy::Preferred);
    setWidget(pBtn);
    connect(pBtn, SIGNAL(clicked()), this, SIGNAL(clicked()));
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
ZenoParamMultilineStr::ZenoParamMultilineStr(const QString &value, LineEditParam param, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_value(value)
    , m_pTextEdit(nullptr)
{
    m_pTextEdit = new QTextEdit;
    setWidget(m_pTextEdit);
    connect(m_pTextEdit, SIGNAL(textChanged()), this, SIGNAL(textChanged()));
    m_pTextEdit->installEventFilter(this);
    m_pTextEdit->setFrameShape(QFrame::NoFrame);
    m_pTextEdit->setFont(param.font);
    m_pTextEdit->setMinimumSize(ZenoStyle::dpiScaledSize(QSize(256, 228)));

	QTextCharFormat format;
    QFont font("HarmonyOS Sans", 12);
	format.setFont(font);
    m_pTextEdit->setCurrentFont(font);
    m_pTextEdit->setText(value);

    QPalette pal = param.palette;
    pal.setColor(QPalette::Base, QColor(37, 37, 37));
    m_pTextEdit->setPalette(pal);
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

void ZenoParamMultilineStr::initTextFormat()
{

}


//////////////////////////////////////////////////////////////////////////////////////
ZenoTextLayoutItem::ZenoTextLayoutItem(const QString &text, const QFont &font, const QColor &color, QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , QGraphicsTextItem(text, parent)
    , m_text(text)
    , m_bRight(false)
{
    setZValue(ZVALUE_ELEMENT);
    setFont(font);
    setDefaultTextColor(color);
    
    setGraphicsItem(this);
    setFlags(ItemSendsScenePositionChanges);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
}

void ZenoTextLayoutItem::setGeometry(const QRectF& geom)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(geom);
    setPos(geom.topLeft());
}

void ZenoTextLayoutItem::setRight(bool right)
{
    m_bRight = right;
}

QRectF ZenoTextLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoTextLayoutItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    //painter->fillRect(boundingRect(), QColor(0, 0, 0));
    if (m_bRight)
    {
        QString text = toPlainText();
        painter->setFont(this->font());
        painter->setPen(QPen(this->defaultTextColor()));
        painter->drawText(boundingRect(), Qt::AlignRight | Qt::AlignCenter, text);
    }
    else
    {
        QGraphicsTextItem::paint(painter, option, widget);
    }
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
    m_minMute = new ZenoImageItem(statusComp.mute, ZenoStyle::dpiScaledSize(QSize(33, 42)), this);
    m_minView = new ZenoImageItem(statusComp.view, ZenoStyle::dpiScaledSize(QSize(25, 42)), this);
    m_minOnce = new ZenoImageItem(statusComp.once, ZenoStyle::dpiScaledSize(QSize(33, 42)), this);
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
    m_minMute->setPos(QPointF(ZenoStyle::dpiScaled(20), 0));
    m_minView->setPos(QPointF(ZenoStyle::dpiScaled(40), 0));

    QSizeF sz2 = m_once->size();
    //todo: kill these magin number.
	QPointF base = QPointF(ZenoStyle::dpiScaled(12), -sz2.height() - ZenoStyle::dpiScaled(8));
	m_once->setPos(base);
	base += QPointF(ZenoStyle::dpiScaled(38), 0);
	m_mute->setPos(base);
	base += QPointF(ZenoStyle::dpiScaled(38), 0);
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
    setChecked(STATUS_ONCE, options & OPT_ONCE);
    setChecked(STATUS_MUTE, options & OPT_MUTE);
    setChecked(STATUS_VIEW, options & OPT_VIEW);
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