#include "statusgroup.h"
#include "statusbutton.h"
#include "zenosvgitem.h"
#include "style/zenostyle.h"
#include "nodeeditor/gv/zgraphicstextitem.h"
#include <QDebug>


static const int skew = 6;  // 倾斜偏移
static const int side = 24; // 底边直线的边长
static const int button_margin = 2; //两个按钮之间的空隙


CommonStatusBtn::CommonStatusBtn(RoundRectInfo info, QGraphicsItem* parent)
    : QGraphicsObject(parent)
    , m_rectInfo(info)
    , m_bOn(false)
    , m_bHovered(false)
    , m_bHasRadiusOnButtom(true)
{
    setAcceptHoverEvents(true);
    setCursor(Qt::PointingHandCursor);
}

void CommonStatusBtn::updateHasRadiusOnButtom(bool bOn) {
    m_bHasRadiusOnButtom = bOn;
}

void CommonStatusBtn::setHovered(bool bHovered) {
    m_bHovered = bHovered;
    update();
}

void CommonStatusBtn::toggle(bool bSelected) {
    if (bSelected == m_bOn)
        return;

    m_bOn = bSelected;
    if (!m_bOn) {
        m_bHovered = false;
    }
    emit toggled(m_bOn);
    update();
}

void CommonStatusBtn::hoverEnterEvent(QGraphicsSceneHoverEvent* event) {
    _base::hoverEnterEvent(event);
    m_bHovered = true;
}

void CommonStatusBtn::hoverMoveEvent(QGraphicsSceneHoverEvent* event) {
    _base::hoverLeaveEvent(event);
}

void CommonStatusBtn::hoverLeaveEvent(QGraphicsSceneHoverEvent* event) {
    _base::hoverLeaveEvent(event);
    m_bHovered = false;
}

void CommonStatusBtn::mousePressEvent(QGraphicsSceneMouseEvent* event) {
    _base::mousePressEvent(event);
    event->setAccepted(true);
}

void CommonStatusBtn::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
    _base::mouseReleaseEvent(event);
    toggle(!m_bOn);
}


ByPassButton::ByPassButton(RoundRectInfo info, QGraphicsItem* parent)
    : CommonStatusBtn(info, parent)
{
    setToolTip(tr("by pass: disable this node and pass through the data"));
}

QRectF ByPassButton::boundingRect() const {
    static const int width = skew + side;
    return QRectF(0, 0, width, m_rectInfo.H);
}

void ByPassButton::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) {
    QPainterPath path;

    const int width = skew + side;
    const int height = m_rectInfo.H;

    path.moveTo(0 + skew, 0);
    path.lineTo(width, 0);
    path.lineTo(width - skew, height);
    path.lineTo(0, height);
    path.closeSubpath();

    QColor fillColor = m_bHovered || m_bOn ? QColor("#FFBD21") : QColor("#2E313A");
    painter->setRenderHint(QPainter::Antialiasing);
    painter->fillPath(path, fillColor);
}


ViewButton::ViewButton(RoundRectInfo info, QGraphicsItem* parent)
    : CommonStatusBtn(info, parent)
{
    setToolTip(tr("view"));
}

QRectF ViewButton::boundingRect() const {
    static const int width = side;
    return QRectF(0, 0, width, m_rectInfo.H);
}

void ViewButton::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) {
    const int radius = m_rectInfo.rtradius;
    const int width = side;
    const int height = m_rectInfo.H;

    QPainterPath path;
    path.moveTo(0 + skew, 0);

    path.lineTo(width - radius, 0);
    path.quadTo(width, 0, width, radius);

    // 右下角
    if (m_bHasRadiusOnButtom) {
        path.lineTo(width, height - radius);
        path.quadTo(width, height, width - radius, height);
    }
    else {
        path.lineTo(width, height);
    }

    path.lineTo(0, height);
    path.closeSubpath();

    QColor fillColor = m_bHovered || m_bOn ? QColor("#00BCD4") : QColor("#2E313A");  // 浅蓝/高亮蓝
    painter->setRenderHint(QPainter::Antialiasing);
    painter->fillPath(path, fillColor);
}

NoCacheButton::NoCacheButton(RoundRectInfo info, QGraphicsItem* parent)
    : CommonStatusBtn(info, parent)
{
    setToolTip(tr("no cache:\n the cache will be removed when the data flow to downstream node"));
}

QRectF NoCacheButton::boundingRect() const {
    static const int width = side;
    return QRectF(0, 0, width, m_rectInfo.H);
}

void NoCacheButton::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) {
    const int radius = m_rectInfo.rtradius;
    const int width = side;
    const int height = m_rectInfo.H;

    QPainterPath path;

    path.moveTo(radius, 0);
    path.lineTo(width, 0);
    path.lineTo(width - skew, m_rectInfo.H);

    if (m_bHasRadiusOnButtom) {
        path.lineTo(width - skew - radius, m_rectInfo.H);
        path.quadTo(0, m_rectInfo.H, 0, m_rectInfo.H - radius);
    }
    else {
        path.lineTo(0, m_rectInfo.H);
    }

    path.lineTo(0, m_rectInfo.ltradius);
    path.quadTo(0, 0, m_rectInfo.ltradius, 0);

    path.closeSubpath();

    QColor fillColor = m_bHovered || m_bOn ? QColor("#FF3300") : QColor("#2E313A");
    painter->setRenderHint(QPainter::Antialiasing);
    painter->fillPath(path, fillColor);
}


ClearSbnCacheButton::ClearSbnCacheButton(RoundRectInfo info, QGraphicsItem* parent)
    : CommonStatusBtn(info, parent)
{
    setToolTip(tr("clear subnet:\n the cache inside the subnet will be removed when the calculation is over"));
}

QRectF ClearSbnCacheButton::boundingRect() const {
    static const int width = skew + side;
    return QRectF(0, 0, width, m_rectInfo.H);  // 宽 60，高 50，实际内容是个倾斜的四边形
}

void ClearSbnCacheButton::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) {
    QPainterPath path;

    const int width = skew + side;
    const int height = m_rectInfo.H;

    path.moveTo(0 + skew, 0);
    path.lineTo(width, 0);
    path.lineTo(width - skew, height);
    path.lineTo(0, height);
    path.closeSubpath();

    QColor fillColor = m_bHovered || m_bOn ? QColor("#E302F8") : QColor("#2E313A");
    painter->setRenderHint(QPainter::Antialiasing);
    painter->fillPath(path, fillColor);
}


LeftStatusBtnGroup::LeftStatusBtnGroup(zeno::NodeType type, RoundRectInfo info, QGraphicsItem* parent)
    : _base(parent)
    , m_rectInfo(info)
    , m_clearsubnet(nullptr)
    , m_nocache(nullptr)
{
    m_nocache = new NoCacheButton(info, this);
    m_nocache->setPos(0, 0);

    connect(m_nocache, &NoCacheButton::toggled, [&](bool hovered) {
        emit toggleChanged(STATUS_NOCACHE, hovered);
        });

    if (type == zeno::Node_SubgraphNode ||
        type == zeno::Node_AssetInstance ||
        type == zeno::Node_AssetReference)
    {
        m_clearsubnet = new ClearSbnCacheButton(info, this);
        m_clearsubnet->setPos(side - skew + button_margin, 0);
        connect(m_clearsubnet, &ClearSbnCacheButton::toggled, [&](bool hovered) {
            emit toggleChanged(STATUS_CLEARSUBNET, hovered);
            });
    }
}

QRectF LeftStatusBtnGroup::boundingRect() const {
    if (m_clearsubnet)
        return QRectF(0, 0, side * 2 + button_margin, m_rectInfo.H);
    else
        return QRectF(0, 0, side + button_margin, m_rectInfo.H);
}

void LeftStatusBtnGroup::setNoCache(bool nocache) {
    if (m_nocache)
        m_nocache->toggle(nocache);
}

void LeftStatusBtnGroup::setClearSubnet(bool clearSbn) {
    if (m_clearsubnet)
        m_clearsubnet->toggle(clearSbn);
}

void LeftStatusBtnGroup::updateRightButtomRadius(bool bHasRadius) {
    if (m_nocache)
        m_nocache->updateHasRadiusOnButtom(bHasRadius);
}


/// <summary>

/// </summary>
RightStatusBtnGroup::RightStatusBtnGroup(RoundRectInfo info, QGraphicsItem* parent)
    : _base(parent)
    , m_rectInfo(info)
    , m_bypass(nullptr)
    , m_view(nullptr)
{
    m_bypass = new ByPassButton(info, this);
    m_view = new ViewButton(info, this);
    m_bypass->setPos(0, 0);
    m_view->setPos(side + button_margin, 0);

    //connect(m_view, SIGNAL(hoverChanged(bool)), m_Imgview, SLOT(setHovered(bool)));
    //connect(m_view, SIGNAL(toggled(bool)), m_Imgview, SLOT(toggle(bool)));
    connect(m_view, &ViewButton::toggled, [&](bool hovered) {
        emit toggleChanged(STATUS_VIEW, hovered);
        });

    connect(m_bypass, &ByPassButton::toggled, [&](bool hovered) {
        emit toggleChanged(STATUS_BYPASS, hovered);
        });
}

QRectF RightStatusBtnGroup::boundingRect() const {
    return QRectF(0, 0, side * 2 + button_margin, m_rectInfo.H);
}

void RightStatusBtnGroup::setView(bool isView) {
    if (m_view)
        m_view->toggle(isView);
}

void RightStatusBtnGroup::setByPass(bool bypass) {
    if (m_bypass)
        m_bypass->toggle(bypass);
}

void RightStatusBtnGroup::updateRightButtomRadius(bool bHasRadius) {
    if (m_view)
        m_view->updateHasRadiusOnButtom(bHasRadius);
}



StatusGroup::StatusGroup(bool bHasOptimStatus, RoundRectInfo info, QGraphicsItem* parent)
    : ZLayoutBackground(parent)
    , m_minOptim(nullptr)
    , m_optim(nullptr)
{
    setColors(false, QColor(0, 0, 0, 0));

    RoundRectInfo rectInfo, roundInfo;
    rectInfo.W = info.W;
    rectInfo.H = info.H;
    roundInfo = info;

    m_minMute = new StatusButton(rectInfo);
    m_minMute->setColor(false, QColor("#E7CD2F"), QColor("#2F3135"));

    m_minView = new StatusButton(roundInfo);
    m_minView->setColor(false, QColor("#26C5C5"), QColor("#2F3135"));

    ZGraphicsLayout* pLayout = new ZGraphicsLayout(true);
    pLayout->setSpacing(1);
    if (bHasOptimStatus) {
        m_minOptim = new StatusButton(rectInfo);
        m_minOptim->setColor(false, QColor("#E302F8"), QColor("#2F3135"));
        pLayout->addItem(m_minOptim);
    }
    pLayout->addItem(m_minMute);
    pLayout->addItem(m_minView);
    this->setLayout(pLayout);

    if (bHasOptimStatus) {
        m_optim = new ZenoImageItem(
            ":/icons/ONCE_dark.svg",
            ":/icons/ONCE_light.svg",
            ":/icons/ONCE_light.svg",
            ZenoStyle::dpiScaledSize(QSize(50, 42)),
            this);
        m_optim->setCheckable(true);
        m_optim->hide();
    }

    m_mute = new ZenoImageItem(
        ":/icons/MUTE_dark.svg",
        ":/icons/MUTE_light.svg",
        ":/icons/MUTE_light.svg",
        ZenoStyle::dpiScaledSize(QSize(50, 42)),
        this);
    m_mute->setCheckable(true);
    m_mute->setZValue(1000);
    m_mute->hide();

    m_view = new ZenoImageItem(
        ":/icons/VIEW_dark.svg",
        ":/icons/VIEW_light.svg",
        ":/icons/VIEW_light.svg",
        ZenoStyle::dpiScaledSize(QSize(50, 42)),
        this);
    m_view->setCheckable(true);
    m_view->setZValue(1000);
    m_view->hide();

    QSizeF sz2 = m_mute->size();
    qreal sMarginTwoBar = 0;
    //todo: kill these magin number.

    QPointF base = QPointF(0 + ZenoStyle::dpiScaled(30), -sz2.height() - sMarginTwoBar);
    if (bHasOptimStatus) {
        m_optim->setPos(base);
        base += QPointF(ZenoStyle::dpiScaled(38), 0);
    }
    m_mute->setPos(base);
    base += QPointF(ZenoStyle::dpiScaled(38), 0);
    m_view->setPos(base);

    if (m_minOptim && m_optim) {
        connect(m_minOptim, SIGNAL(hoverChanged(bool)), m_optim, SLOT(setHovered(bool)));
        connect(m_minOptim, SIGNAL(toggled(bool)), m_optim, SLOT(toggle(bool)));
        connect(m_optim, SIGNAL(hoverChanged(bool)), m_minOptim, SLOT(setHovered(bool)));
        connect(m_optim, SIGNAL(toggled(bool)), m_minOptim, SLOT(toggle(bool)));
        connect(m_minOptim, &StatusButton::toggled, [=](bool hovered) {
            emit toggleChanged(STATUS_OWNING, hovered);
            });
    }

    connect(m_minView, SIGNAL(hoverChanged(bool)), m_view, SLOT(setHovered(bool)));
    connect(m_minView, SIGNAL(toggled(bool)), m_view, SLOT(toggle(bool)));
    connect(m_view, SIGNAL(hoverChanged(bool)), m_minView, SLOT(setHovered(bool)));
    connect(m_view, SIGNAL(toggled(bool)), m_minView, SLOT(toggle(bool)));
    connect(m_minView, &StatusButton::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_VIEW, hovered);
        });

    connect(m_minMute, SIGNAL(hoverChanged(bool)), m_mute, SLOT(setHovered(bool)));
    connect(m_minMute, SIGNAL(toggled(bool)), m_mute, SLOT(toggle(bool)));
    connect(m_mute, SIGNAL(hoverChanged(bool)), m_minMute, SLOT(setHovered(bool)));
    connect(m_mute, SIGNAL(toggled(bool)), m_minMute, SLOT(toggle(bool)));
    connect(m_minMute, &StatusButton::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_BYPASS, hovered);
        });

    setAcceptHoverEvents(true);
}

void StatusGroup::updateRightButtomRadius(bool bHasRadius) {
    m_minView->updateRightButtomRadius(bHasRadius);
}

QRectF StatusGroup::boundingRect() const
{
    return _base::boundingRect();
}

void StatusGroup::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    _base::paint(painter, option, widget);
}

void StatusGroup::setChecked(STATUS_BTN btn, bool bChecked)
{

}

void StatusGroup::setOptions(int options)
{

}

void StatusGroup::setView(bool isView)
{
    m_view->toggle(isView);
}

void StatusGroup::onZoomed()
{
    if (1 - editor_factor > 0.00001f)
    {
        QSize size = QSize(ZenoStyle::scaleWidth(50), ZenoStyle::scaleWidth(42));
        if (m_optim)
            m_optim->resize(size);
        m_mute->resize(size);
        m_view->resize(size);

        QSizeF sz2 = m_mute->size();
        qreal sMarginTwoBar = ZenoStyle::dpiScaled(0);
        QPointF base = QPointF(ZenoStyle::scaleWidth(0), -sz2.height() - sMarginTwoBar);
        qreal offset = ZenoStyle::scaleWidth(38);
        if (m_optim) {
            base += QPointF(offset, 0);
            m_optim->setPos(base);
        }
        base += QPointF(offset, 0);
        m_mute->setPos(base);
        base += QPointF(offset, 0);
        m_view->setPos(base);
    }
}

void StatusGroup::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    m_mute->show();
    m_view->show();
    if (m_optim)
        m_optim->show();
    _base::hoverEnterEvent(event);
}

void StatusGroup::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void StatusGroup::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    if (!m_mute->isHovered())
        m_mute->hide();
    if (!m_view->isHovered())
        m_view->hide();
    if (m_optim) {
        if (!m_optim->isHovered())
            m_optim->hide();
    }
    _base::hoverLeaveEvent(event);
}
