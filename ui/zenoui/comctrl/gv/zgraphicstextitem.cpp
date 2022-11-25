#include "zgraphicstextitem.h"
#include "zenosocketitem.h"
#include "render/common_id.h"
#include <zenomodel/include/modelrole.h>
#include "zassert.h"
#include <zenoui/style/zenostyle.h>


qreal editor_factor = 1.0;


ZGraphicsTextItem::ZGraphicsTextItem(QGraphicsItem* parent)
    : QGraphicsTextItem(parent)
{
}

ZGraphicsTextItem::ZGraphicsTextItem(const QString& text, const QFont& font, const QColor& color, QGraphicsItem* parent)
    : QGraphicsTextItem(parent)
{
    setText(text);
}

void ZGraphicsTextItem::setText(const QString& text)
{
    m_text = text;
    setPlainText(m_text);
}

void ZGraphicsTextItem::setMargins(qreal leftM, qreal topM, qreal rightM, qreal bottomM)
{
    QTextFrame* frame = document()->rootFrame();
    QTextFrameFormat format = frame->frameFormat();
    format.setLeftMargin(leftM);
    format.setRightMargin(rightM);
    format.setTopMargin(topM);
    format.setBottomMargin(bottomM);
    frame->setFrameFormat(format);
}

void ZGraphicsTextItem::setBackground(const QColor& clr)
{
    m_bg = clr;
}

QRectF ZGraphicsTextItem::boundingRect() const
{
    return QGraphicsTextItem::boundingRect();
}

void ZGraphicsTextItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    QStyleOptionGraphicsItem myOption(*option);
    myOption.state &= ~QStyle::State_Selected;
    myOption.state &= ~QStyle::State_HasFocus;
    if (m_bg.isValid())
    {
        painter->setPen(Qt::NoPen);
        painter->setBrush(m_bg);
        painter->drawRect(boundingRect());
    }
    QGraphicsTextItem::paint(painter, option, widget);
}

QPainterPath ZGraphicsTextItem::shape() const
{
    //ensure the hit test was overrided by whole rect, rather than text region.
    QPainterPath path;
    path.addRect(boundingRect());
    return path;
}

void ZGraphicsTextItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsTextItem::mousePressEvent(event);
}

void ZGraphicsTextItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsTextItem::mouseMoveEvent(event);
}

void ZGraphicsTextItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsTextItem::mouseReleaseEvent(event);
}

void ZGraphicsTextItem::focusOutEvent(QFocusEvent* event)
{
    QGraphicsTextItem::focusOutEvent(event);
    emit editingFinished();

    QString newText = document()->toPlainText();
    if (newText != m_text)
    {
        QString oldText = m_text;
        m_text = newText;
        emit contentsChanged(oldText, newText);
    }
}



ZSimpleTextItem::ZSimpleTextItem(QGraphicsItem* parent)
    : base(parent)
    , m_fixedWidth(-1)
    , m_bRight(false)
    , m_bHovered(false)
    , m_alignment(Qt::AlignLeft)
    , m_hoverCursor(Qt::ArrowCursor)
{
    setFlags(ItemIsFocusable | ItemIsSelectable);
    setAcceptHoverEvents(true);
}

ZSimpleTextItem::ZSimpleTextItem(const QString& text, QGraphicsItem* parent)
    : base(text, parent)
    , m_fixedWidth(-1)
    , m_bRight(false)
    , m_bHovered(false)
    , m_alignment(Qt::AlignLeft)
{
    setAcceptHoverEvents(true);
    updateBoundingRect();
}

ZSimpleTextItem::~ZSimpleTextItem()
{

}

QRectF ZSimpleTextItem::boundingRect() const
{
    return m_boundingRect;
}

QPainterPath ZSimpleTextItem::shape() const
{
    QPainterPath path;
    path.addRect(boundingRect());
    return path;
}

void ZSimpleTextItem::updateBoundingRect()
{
    QTextLayout layout(text(), font());
    QSizeF sz = size(text(), font(), m_padding.left, m_padding.top, m_padding.right, m_padding.bottom);
    if (m_fixedWidth > 0)
    {
        m_boundingRect = QRectF(0, 0, m_fixedWidth, sz.height());
    }
    else
    {
        m_boundingRect = QRectF(0, 0, sz.width(), sz.height());
    }
}

void ZSimpleTextItem::setPadding(int left, int top, int right, int bottom)
{
    m_padding.left = left;
    m_padding.right = right;
    m_padding.top = top;
    m_padding.bottom = bottom;
    updateBoundingRect();
}

void ZSimpleTextItem::setAlignment(Qt::Alignment align)
{
    m_alignment = align;
}

void ZSimpleTextItem::setFixedWidth(qreal fixedWidth)
{
    m_fixedWidth = fixedWidth;
    updateBoundingRect();
}

QRectF ZSimpleTextItem::setupTextLayout(QTextLayout* layout, _padding padding, Qt::Alignment align, qreal fixedWidth)
{
    layout->setCacheEnabled(true);
    layout->beginLayout();
    while (layout->createLine().isValid())
        ;
    layout->endLayout();
    qreal maxWidth = 0;
    qreal y = 0;
    for (int i = 0; i < layout->lineCount(); ++i) {
        QTextLine line = layout->lineAt(i);
        qreal wtf = line.width();
        maxWidth = qMax(maxWidth, line.naturalTextWidth() + padding.left + padding.right);

        qreal x = 0;
        qreal w = line.horizontalAdvance();
        if (fixedWidth > 0)
        {
            if (align == Qt::AlignCenter)
            {
                x = (fixedWidth - w) / 2;
            }
            else if (align == Qt::AlignRight)
            {
                x = (fixedWidth - w);
            }
        }
        line.setPosition(QPointF(x, y + padding.top));
        y += line.height() + padding.top + padding.bottom;
    }
    return QRectF(0, 0, maxWidth, y);
}

void ZSimpleTextItem::setBackground(const QColor& clr)
{
    m_bg = clr;
}

void ZSimpleTextItem::setText(const QString& text)
{
    base::setText(text);
    updateBoundingRect();
}

void ZSimpleTextItem::setHoverCursor(Qt::CursorShape cursor)
{
    m_hoverCursor = cursor;
}

void ZSimpleTextItem::setRight(bool right)
{
    m_bRight = right;
}

bool ZSimpleTextItem::isHovered() const
{
    return m_bHovered;
}

QSizeF ZSimpleTextItem::size(const QString& text, const QFont& font, int pleft, int pTop, int pRight, int pBottom)
{
    QTextLayout layout(text, font);
    QRectF rc = setupTextLayout(&layout, _padding(pleft, pTop, pRight, pBottom));
    return rc.size();
}

void ZSimpleTextItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    m_bHovered = true;
    base::hoverEnterEvent(event);
    setCursor(m_hoverCursor);
}

void ZSimpleTextItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    base::hoverMoveEvent(event);
}

void ZSimpleTextItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    base::hoverLeaveEvent(event);
    m_bHovered = false;
    setCursor(Qt::ArrowCursor);
}

void ZSimpleTextItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    base::mousePressEvent(event);
}

void ZSimpleTextItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    base::mouseMoveEvent(event);
}

void ZSimpleTextItem::keyPressEvent(QKeyEvent* event)
{
    base::keyPressEvent(event);
}

void ZSimpleTextItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
#if 0
    if (editor_factor < 0.2)
        return;
#endif
    if (m_bg.isValid())
    {
        painter->save();
        painter->setPen(Qt::NoPen);
        painter->setBrush(m_bg);
        painter->drawRect(boundingRect());
        painter->restore();
    }

    painter->setFont(this->font());

    QString tmp = text();
    tmp.replace(QLatin1Char('\n'), QChar::LineSeparator);
    QTextLayout layout(tmp, font());

    QPen p;
    if (option->state & QStyle::State_MouseOver)
    {
        p.setBrush(QColor(255,255,255));
    }
    else
    {
        p.setBrush(brush());
    }

    painter->setPen(p);
    if (pen().style() == Qt::NoPen && brush().style() == Qt::SolidPattern) {
        painter->setBrush(Qt::NoBrush);
    }
    else {
        QTextLayout::FormatRange range;
        range.start = 0;
        range.length = layout.text().length();
        range.format.setTextOutline(pen());
        layout.setFormats(QVector<QTextLayout::FormatRange>(1, range));
    }

    qreal w = boundingRect().width();
    setupTextLayout(&layout, m_padding, m_alignment, m_fixedWidth == -1 ? w : m_fixedWidth);

    layout.draw(painter, QPointF(0, 0));
}


ZSocketGroupItem::ZSocketGroupItem(
        const QPersistentModelIndex& viewSockIdx,
        const QString& sockName, 
        bool bInput,
        Callback_OnSockClicked cbSockOnClick,
        QGraphicsItem* parent
    )
    : ZSimpleTextItem(sockName, parent)
    , m_bInput(bInput)
    , m_viewSockIdx(viewSockIdx)
{
    ZASSERT_EXIT(m_viewSockIdx.isValid());

    ImageElement elem;
    elem.image = ":/icons/socket-off.svg";
    elem.imageHovered = ":/icons/socket-hover.svg";
    elem.imageOn = ":/icons/socket-on.svg";
    elem.imageOnHovered = ":/icons/socket-on-hover.svg";

    m_socket = new ZenoSocketItem(viewSockIdx, sockName, bInput, elem, ZenoStyle::dpiScaledSize(QSizeF(cSocketWidth, cSocketHeight)), this);
    QObject::connect(m_socket, &ZenoSocketItem::clicked, [=]() {
        cbSockOnClick(m_socket);
    });

    setBrush(QColor(188, 188, 188));
    QFont font("HarmonyOS Sans Bold", 11);
    font.setBold(true);
    setFont(font);
    updateBoundingRect();

    const QString& coreType = viewSockIdx.data(ROLE_PARAM_TYPE).toString();
    if (bInput && 
        (coreType == "list" ||
        coreType == "dict" ||
        coreType == "DictObject:NumericObject"))
    {
        QRectF rc = this->boundingRect();

        ImageElement elemSwitch;
        elemSwitch.image = ":/icons/checkbox_unchecked.svg";
        elemSwitch.imageHovered = ":/icons/checkbox_unchecked.svg";
        elemSwitch.imageOn = ":/icons/checkbox_checked_cyan.svg";
        elemSwitch.imageOnHovered = ":/icons/checkbox_checked_cyan.svg";

        PARAM_CONTROL ctrl = (PARAM_CONTROL)m_viewSockIdx.data(ROLE_PARAM_CTRL).toInt();

        ZenoImageItem* pSwitch = new ZenoImageItem(elemSwitch, ZenoStyle::dpiScaledSize(QSizeF(20, 20)), this);
        pSwitch->setPos(rc.width() + 10, 0);
        pSwitch->setCheckable(true);
        pSwitch->toggle(ctrl == CONTROL_DICTPANEL);
        QObject::connect(pSwitch, &ZenoImageItem::toggled, [=](bool bChecked) {
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_viewSockIdx.model());
            pModel->setData(m_viewSockIdx, bChecked ? CONTROL_DICTPANEL : CONTROL_NONE, ROLE_PARAM_CTRL);
        });
    }

    setFlag(ItemSendsGeometryChanges);
    setFlag(ItemSendsScenePositionChanges);
}

ZenoSocketItem* ZSocketGroupItem::socketItem() const
{
    return m_socket;
}

QPointF ZSocketGroupItem::getPortPos()
{
    return m_socket->sceneBoundingRect().center();
}

bool ZSocketGroupItem::getSocketInfo(bool& bInput, QString& nodeid, QString& sockName)
{
    return m_socket->getSocketInfo(bInput, nodeid, sockName);
}

QVariant ZSocketGroupItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
    if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        //adjust the pos of socket.
        //the parent of this item pos.
        QGraphicsItem* parent = this->parentItem();
        if (parent) {
            QRectF br = parent->sceneBoundingRect();
            qreal x = 0, y = 0;
            y = boundingRect().height() / 2 - cSocketHeight / 2;
            static int sBorder = 2;
            if (m_bInput) {
                x = mapFromScene(br.topLeft()).x() - cSocketWidth / 2 + sBorder;
            }
            else {
                x = mapFromScene(br.bottomRight()).x() - cSocketWidth / 2 - sBorder;
            }
            m_socket->setPos(QPointF(x, y));
        }
    }
    return value;
}


ZSocketEditableItem::ZSocketEditableItem(
        const QPersistentModelIndex& viewSockIdx,
        const QString& sockName,
        bool bInput,
        Callback_OnSockClicked cbSockOnClick,
        Callback_EditContentsChange cbSockRename,
        QGraphicsItem* parent
    )
    : _base(parent)
    , m_bInput(bInput)
    , m_viewSockIdx(viewSockIdx)
{
    setText(sockName);

    ImageElement elem;
    elem.image = ":/icons/socket-off.svg";
    elem.imageHovered = ":/icons/socket-hover.svg";
    elem.imageOn = ":/icons/socket-on.svg";
    elem.imageOnHovered = ":/icons/socket-on-hover.svg";

    m_socket = new ZenoSocketItem(viewSockIdx, sockName, bInput, elem, QSizeF(cSocketWidth, cSocketHeight), this);
    m_socket->setZValue(ZVALUE_ELEMENT);
    QObject::connect(m_socket, &ZenoSocketItem::clicked, [=]() {
        cbSockOnClick(m_socket);
    });

    setDefaultTextColor(QColor(188, 188, 188));
    QFont font("HarmonyOS Sans Bold", 11);
    font.setBold(true);
    setFont(font);

    setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    //setData(GVKEY_SIZEHINT, QSizeF(32, 32));
    setTextInteractionFlags(Qt::TextEditorInteraction);

    QTextFrame* frame = document()->rootFrame();
    QTextFrameFormat format = frame->frameFormat();
    format.setBackground(QColor(37, 37, 37));
    frame->setFrameFormat(format);

    setFlag(ItemSendsGeometryChanges);
    setFlag(ItemSendsScenePositionChanges);

    QObject::connect(this, &ZSocketEditableItem::contentsChanged, cbSockRename);
}

void ZSocketEditableItem::updateSockName(const QString& name)
{
    setPlainText(name);

    //have to reset the text format, which is trivial.
    QTextFrame* frame = document()->rootFrame();
    QTextFrameFormat format = frame->frameFormat();
    format.setBackground(QColor(37, 37, 37));
    frame->setFrameFormat(format);
}

QPointF ZSocketEditableItem::getPortPos()
{
    return m_socket->sceneBoundingRect().center();
}

ZenoSocketItem* ZSocketEditableItem::socketItem() const
{
    return m_socket;
}

bool ZSocketEditableItem::getSocketInfo(bool& bInput, QString& nodeid, QString& sockName)
{
    //todo: get from m_paramIdx;
    return m_socket->getSocketInfo(bInput, nodeid, sockName);
}

QVariant ZSocketEditableItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
    if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        //adjust the pos of socket.
        //the parent of this item pos.
        QGraphicsItem* parent = parentItem();
        if (parent) {
            QRectF br = parent->sceneBoundingRect();
            qreal x = 0, y = 0;
            y = boundingRect().height() / 2 - cSocketHeight / 2;
            static int sBorder = 2;
            if (m_bInput) {
                x = mapFromScene(br.topLeft()).x() - cSocketWidth / 2 + sBorder;
            }
            else {
                x = mapFromScene(br.bottomRight()).x() - cSocketWidth / 2 - sBorder;
            }
            m_socket->setPos(QPointF(x, y));
        }
    }
    return value;
}

void ZSocketEditableItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    QStyleOptionGraphicsItem myOption(*option);
    myOption.state &= ~QStyle::State_Selected;
    myOption.state &= ~QStyle::State_HasFocus;
    _base::paint(painter, &myOption, widget);
}
