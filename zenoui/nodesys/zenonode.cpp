#include "zenonode.h"
#include "../model/modelrole.h"
#include "../model/subgraphmodel.h"
#include "../render/common_id.h"
#include "zenoparamnameitem.h"
#include "zenoparamwidget.h"
#include "zenoheatmapitem.h"
#include "util/uihelper.h"


ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_bInitSockets(false)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
    , m_bCollasped(false)
    , m_collaspedWidget(nullptr)
    , m_bHeapMap(false)
    , m_pMainLayout(nullptr)
{
    setFlags(ItemIsMovable | ItemIsSelectable);
}

ZenoNode::~ZenoNode()
{
}

void ZenoNode::_updateSocketItemPos()
{
    //need to optimizize
    for (auto sockName : m_inSockNames.keys())
    {
        auto sockLabelItem = m_inSockNames[sockName];
        auto socketItem = m_inSocks[sockName];
        QPointF scenePos = sockLabelItem->scenePos();
        QRectF sRect = sockLabelItem->sceneBoundingRect();
        QPointF pos = this->mapFromScene(scenePos);
        qreal x = -socketItem->size().width() / 2;
        qreal y = pos.y() + sRect.height() / 2 - socketItem->size().height() / 2;
        pos -= QPointF(m_renderParams.socketToText + socketItem->size().width(), 0);
        //fixed center on the border.
        //socket icon is hard to layout, as it's not a part of layout item but rely on the position of the layouted item.
        pos.setX(-socketItem->size().width() / 2);
        pos.setY(y);

        socketItem->setPos(pos);
    }
    for (auto sockName : m_outSockNames.keys())
    {
        auto sockLabelItem = m_outSockNames[sockName];
        auto socketItem = m_outSocks[sockName];
        QRectF sRect = sockLabelItem->sceneBoundingRect();
        QPointF scenePos = sRect.topRight();
        sRect = mapRectFromScene(sRect);
        QPointF pos;

        int x = m_bodyWidget->rect().width() - socketItem->size().width() / 2;
        int y = sRect.center().y() - socketItem->size().height() / 2;
        pos.setX(x);
        pos.setY(y);

        socketItem->setPos(pos);
    }
}

void ZenoNode::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    //need to init socket icon pos elsewhere.
    _updateSocketItemPos();
    if (isSelected())
    {
        //draw border
        painter->setRenderHint(QPainter::Antialiasing, true);
        QPen pen(QColor(238, 136, 68), 3);
        painter->setPen(pen);
        QRectF rc = m_pMainLayout->geometry();
        qreal offset = 1;
        rc = rc.adjusted(-offset, -offset, offset, offset);
        QPainterPath path = UiHelper::getRoundPath(rc, m_renderParams.headerBg.lt_radius, m_renderParams.headerBg.rt_radius, m_renderParams.bodyBg.lb_radius, m_renderParams.bodyBg.rb_radius, true);
        painter->drawPath(path);
    }
}

QRectF ZenoNode::boundingRect() const
{
    return childrenBoundingRect();
}

int ZenoNode::type() const
{
    return Type;
}

void ZenoNode::init(const QModelIndex& index)
{
    m_index = QPersistentModelIndex(index);

    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_OBJTYPE).toInt());

    m_collaspedWidget = initCollaspedWidget();
    m_collaspedWidget->setVisible(m_bCollasped);

    m_headerWidget = initHeaderBgWidget(type);
    if (type != BLACKBOARD_NODE) {
        initIndependentWidgets();
    }
    m_bodyWidget = initBodyWidget(type);

    m_pMainLayout = new QGraphicsLinearLayout(Qt::Vertical);
    m_pMainLayout->addItem(m_collaspedWidget);
    m_pMainLayout->addItem(m_headerWidget);
    m_pMainLayout->addItem(m_bodyWidget);
    m_pMainLayout->setContentsMargins(0, 0, 0, 0);
    m_pMainLayout->setSpacing(0);

    setLayout(m_pMainLayout);

    if (type == BLACKBOARD_NODE)
    {
        setZValue(ZVALUE_BLACKBOARD);
    }

    QPointF pos = m_index.data(ROLE_OBJPOS).toPointF();
    const QString &id = m_index.data(ROLE_OBJID).toString();
    setPos(pos);

    // setPos will send geometry, but it's not supposed to happend during initialization.
    setFlag(ItemSendsGeometryChanges);
    setFlag(ItemSendsScenePositionChanges);

    const SubGraphModel *pModel = qobject_cast<const SubGraphModel *>(m_index.model());
    Q_ASSERT(pModel);
    connect(this, SIGNAL(doubleClicked(const QString&)), pModel, SLOT(onDoubleClicked(const QString&)));
}

void ZenoNode::initIndependentWidgets()
{
    QRectF rc;

    rc = m_renderParams.rcMute;
    m_mute = new ZenoImageItem(m_renderParams.mute, QSizeF(rc.width(), rc.height()), this);
    m_mute->setPos(rc.topLeft());
    m_mute->setZValue(ZVALUE_ELEMENT);

    rc = m_renderParams.rcView;
    m_view = new ZenoImageItem(m_renderParams.view, QSizeF(rc.width(), rc.height()), this);
    m_view->setPos(rc.topLeft());
    m_view->setZValue(ZVALUE_ELEMENT);

    rc = m_renderParams.rcPrep;
    m_prep = new ZenoImageItem(m_renderParams.prep, QSizeF(rc.width(), rc.height()), this);
    m_prep->setPos(rc.topLeft());
    m_prep->setZValue(ZVALUE_ELEMENT);

    rc = m_renderParams.rcCollasped;
    m_collaspe = new ZenoImageItem(m_renderParams.collaspe, QSizeF(rc.width(), rc.height()), this);
    m_collaspe->setPos(rc.topLeft());
    m_collaspe->setZValue(ZVALUE_ELEMENT);
    connect(m_collaspe, SIGNAL(clicked()), this, SLOT(onCollaspeBtnClicked()));
}

ZenoBackgroundWidget* ZenoNode::initCollaspedWidget()
{
    ZenoBackgroundWidget *widget = new ZenoBackgroundWidget(this);
    widget->setRadius(10, 10, 10, 10);
    const auto &headerBg = m_renderParams.headerBg;
    widget->setColors(headerBg.bAcceptHovers, headerBg.clr_normal, headerBg.clr_hovered, headerBg.clr_selected);

    QGraphicsLinearLayout *pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    const QString &name = m_index.data(ROLE_OBJNAME).toString();
    QFont font = m_renderParams.nameFont;
    font.setPointSize(font.pointSize() + 4);
    ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem(name, font, m_renderParams.nameClr.color());
    pNameItem->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    int horizontalPadding = 20;

    pHLayout->addStretch();
    pHLayout->addItem(new SpacerLayoutItem(QSizeF(horizontalPadding, 6), true));
    pHLayout->addItem(pNameItem);
    pHLayout->addItem(new SpacerLayoutItem(QSizeF(horizontalPadding, 6), true));
    pHLayout->setAlignment(pNameItem, Qt::AlignCenter);
    pHLayout->addStretch();

    widget->setLayout(pHLayout);
    return widget;
}

ZenoBackgroundWidget *ZenoNode::initHeaderBgWidget(NODE_TYPE type)
{
    ZenoBackgroundWidget* headerWidget = new ZenoBackgroundWidget(this);

    const auto &headerBg = m_renderParams.headerBg;
    headerWidget->setRadius(headerBg.lt_radius, headerBg.rt_radius, headerBg.lb_radius, headerBg.rb_radius);
    headerWidget->setColors(headerBg.bAcceptHovers, headerBg.clr_normal, headerBg.clr_hovered, headerBg.clr_selected);

    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    const QString &name = m_index.data(ROLE_OBJNAME).toString();
    ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem(name, m_renderParams.nameFont, m_renderParams.nameClr.color());
    pNameItem->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    QFontMetrics metrics(m_renderParams.nameFont);
    int textWidth = metrics.horizontalAdvance(name);
    int horizontalPadding = 20;

    QRectF rc = m_renderParams.rcCollasped;

    ZenoSvgLayoutItem *collaspeItem = new ZenoSvgLayoutItem(m_renderParams.collaspe, QSizeF(rc.width(), rc.height()));
    collaspeItem->setZValue(ZVALUE_ELEMENT);
    collaspeItem->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    connect(collaspeItem, SIGNAL(clicked()), this, SLOT(onCollaspeBtnClicked()));

    pHLayout->addStretch();
    pHLayout->addItem(pNameItem);
    pHLayout->setAlignment(pNameItem, Qt::AlignCenter);
    pHLayout->addStretch();

    pHLayout->setContentsMargins(0, 5, 0, 5);
    pHLayout->setSpacing(0);

    headerWidget->setLayout(pHLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);
    if (type == BLACKBOARD_NODE)
    {
        QColor clr(98, 108, 111);
        headerWidget->setColors(false, clr, clr, clr);
    }
    return headerWidget;
}

ZenoBackgroundWidget* ZenoNode::initBodyWidget(NODE_TYPE type)
{
    ZenoBackgroundWidget *bodyWidget = new ZenoBackgroundWidget(this);

    const auto &bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(bodyBg.bAcceptHovers, bodyBg.clr_normal, bodyBg.clr_hovered, bodyBg.clr_selected);

    QGraphicsLinearLayout *pVLayout = new QGraphicsLinearLayout(Qt::Vertical);

    if (type != BLACKBOARD_NODE)
    {
        if (QGraphicsGridLayout *pParamsLayout = initParams()) {
            pParamsLayout->setContentsMargins(m_renderParams.distParam.paramsLPadding, 10, 10, 0);
            pVLayout->addItem(pParamsLayout);
        }
        if (QGraphicsGridLayout *pSocketsLayout = initSockets()) {
            pSocketsLayout->setContentsMargins(m_renderParams.distParam.paramsLPadding, m_renderParams.distParam.paramsToTopSocket, m_renderParams.distParam.paramsLPadding, 0);
            pVLayout->addItem(pSocketsLayout);
        }

        //heapmap stays at the bottom of node layout.
        COLOR_RAMPS ramps = m_index.data(ROLE_COLORRAMPS).value<COLOR_RAMPS>();
        if (!ramps.isEmpty()) {
            ZenoHeatMapItem *pItem = new ZenoHeatMapItem(ramps);
            pVLayout->addItem(pItem);
        }
        bodyWidget->setZValue(ZVALUE_ELEMENT);
    }
    else
    {
        QColor clr(0, 0, 0);
        bodyWidget->setColors(false, clr, clr, clr);
        const QSize &sz = m_index.data(ROLE_BLACKBOARD_SIZE).toSize();
        const QString& content = m_index.data(ROLE_BLACKBOARD_CONTENT).toString();

        ZenoBoardTextLayoutItem* pTextItem = new ZenoBoardTextLayoutItem(content, m_renderParams.boardFont, m_renderParams.boardTextClr.color(), sz);
        //pVLayout->addStretch();
        pVLayout->addItem(pTextItem);
    }

    bodyWidget->setLayout(pVLayout);
    return bodyWidget;
}

QGraphicsGridLayout* ZenoNode::initParams()
{
    const PARAMS_INFO &params = m_index.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    QList<QString> names = params.keys();
    int r = 0, n = names.length();

    QGraphicsGridLayout *pParamsLayout = nullptr;
    if (n > 0)
    {
        pParamsLayout = new QGraphicsGridLayout;
        for (auto paramName : params.keys())
        {
            const PARAM_INFO &param = params[paramName];
            if (param.bEnableConnect)
                continue;

            QVariant val = param.value;
            QString value;
            if (val.type() == QVariant::String)
            {
                value = val.toString();
            }
            else if (val.type() == QVariant::Double)
            {
                value = QString::number(val.toDouble());
            }

            ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem(paramName, m_renderParams.paramFont, m_renderParams.paramClr.color());
            pParamsLayout->addItem(pNameItem, r, 0);

            switch (param.control)
            {
                case CONTROL_STRING:
                case CONTROL_INT:
                case CONTROL_FLOAT:
                case CONTROL_BOOL:
                {
                    ZenoParamLineEdit *pLineEdit = new ZenoParamLineEdit(value, m_renderParams.lineEditParam);
                    pParamsLayout->addItem(pLineEdit, r, 1);
                    break;
                }
                case CONTROL_ENUM:
                {
                    QStringList items = param.typeDesc.mid(QString("enum ").length()).split(QRegExp("\\s+"));
                    ZenoParamComboBox *pComboBox = new ZenoParamComboBox(items, m_renderParams.comboboxParam);
                    pParamsLayout->addItem(pComboBox, r, 1);
                    break;
                }
                case CONTROL_READPATH:
                {
                    ZenoParamLineEdit *pFileWidget = new ZenoParamLineEdit(value, m_renderParams.lineEditParam);
                    ZenoParamPushButton* pBtn = new ZenoParamPushButton("...");
                    pParamsLayout->addItem(pFileWidget, r, 1);
                    pParamsLayout->addItem(pBtn, r, 2);
                    break;
                }
                case CONTROL_WRITEPATH:
                {
                    ZenoParamLineEdit *pFileWidget = new ZenoParamLineEdit(value, m_renderParams.lineEditParam);
                    ZenoParamPushButton *pBtn = new ZenoParamPushButton("...");
                    pParamsLayout->addItem(pFileWidget, r, 1);
                    pParamsLayout->addItem(pBtn, r, 2);
                    break;
                }
                case CONTROL_MULTILINE_STRING:
                {
                    ZenoParamMultilineStr *pMultiStrEdit = new ZenoParamMultilineStr(value);
                    pParamsLayout->addItem(pMultiStrEdit, ++r, 0);
                    break;
                }
                case CONTROL_HEAPMAP:
                {
                    m_bHeapMap = true;
                    //break;
                }
                default:
                {
                    ZenoTextLayoutItem *pValueItem = new ZenoTextLayoutItem(value, m_renderParams.paramFont, m_renderParams.paramClr.color());
                    pParamsLayout->addItem(pValueItem, r, 1);
                    break;
                }
            }
            r++;
        }
    }
    return pParamsLayout;
}

QGraphicsGridLayout* ZenoNode::initSockets()
{
    const QString &nodeid = nodeId();
    QGraphicsGridLayout *pSocketsLayout = new QGraphicsGridLayout;
    {
        INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        int r = 0;
        for (auto inSock : inputs.keys()) {
            ZenoSocketItem *socket = new ZenoSocketItem(SOCKET_INFO(nodeid, inSock, QPointF(), true), m_renderParams.socket, m_renderParams.szSocket, this);
            m_inSocks.insert(std::make_pair(inSock, socket));
            socket->setZValue(ZVALUE_ELEMENT);

            ZenoTextLayoutItem *pSocketItem = new ZenoTextLayoutItem(inSock, m_renderParams.paramFont, m_renderParams.paramClr.color());
            pSocketsLayout->addItem(pSocketItem, r, 0);
            //connect(pSocketItem, SIGNAL(geometrySetup(const QPointF &)), socket, SLOT(socketNamePosition(const QPointF &)));
            m_inSockNames.insert(inSock, pSocketItem);

            r++;
        }
        OUTPUT_SOCKETS outputs = m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        for (auto outSock : outputs.keys())
        {
            ZenoTextLayoutItem *pSocketItem = new ZenoTextLayoutItem(outSock, m_renderParams.paramFont, m_renderParams.paramClr.color());

            QGraphicsLinearLayout *pMiniLayout = new QGraphicsLinearLayout(Qt::Horizontal);
            pMiniLayout->addStretch();
            pMiniLayout->addItem(pSocketItem);
            pSocketsLayout->addItem(pMiniLayout, r, 1);

            m_outSockNames.insert(outSock, pSocketItem);

            ZenoSocketItem *socket = new ZenoSocketItem(SOCKET_INFO(nodeid, outSock, QPointF(), false), m_renderParams.socket, m_renderParams.szSocket, this);
            m_outSocks.insert(std::make_pair(outSock, socket));
            socket->setZValue(ZVALUE_ELEMENT);
            //connect(pSocketItem, SIGNAL(geometrySetup(const QPointF &)), socket, SLOT(socketNamePosition(const QPointF &)));

            r++;
        }
    }
    return pSocketsLayout;
}

void ZenoNode::toggleSocket(bool bInput, const QString& sockName, bool bSelected)
{
    if (bInput) {
        auto itPort = m_inSocks.find(sockName);
        Q_ASSERT(itPort != m_inSocks.end());
        itPort->second->toggle(bSelected);
    } else {
        auto itPort = m_outSocks.find(sockName);
        Q_ASSERT(itPort != m_outSocks.end());
        itPort->second->toggle(bSelected);
    }
}

QPointF ZenoNode::getPortPos(bool bInput, const QString &portName)
{
    if (m_bCollasped)
    {
        m_collaspedWidget->show();
        QRectF rc = m_collaspedWidget->sceneBoundingRect();
        if (bInput)
        {
            return QPointF(rc.left(), rc.center().y());
        }
        else
        {
            return QPointF(rc.right(), rc.center().y());
        }
    }
    else
    {
        QString id = nodeId();
        if (bInput) {
            auto itPort = m_inSocks.find(portName);
            Q_ASSERT(itPort != m_inSocks.end());
            QPointF pos = itPort->second->sceneBoundingRect().center();
            return pos;
        } else {
            auto itPort = m_outSocks.find(portName);
            Q_ASSERT(itPort != m_outSocks.end());
            QPointF pos = itPort->second->sceneBoundingRect().center();
            return pos;
        }
    }
}

QString ZenoNode::nodeId() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OBJID).toString();
}

QString ZenoNode::nodeName() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OBJNAME).toString();
}

QPointF ZenoNode::nodePos() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OBJPOS).toPointF();
}

INPUT_SOCKETS ZenoNode::inputParams() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
}

OUTPUT_SOCKETS ZenoNode::outputParams() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
}

bool ZenoNode::sceneEventFilter(QGraphicsItem* watched, QEvent* event)
{
    return _base::sceneEventFilter(watched, event);
}

bool ZenoNode::sceneEvent(QEvent *event)
{
    return _base::sceneEvent(event);
}

void ZenoNode::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseDoubleClickEvent(event);
    emit doubleClicked(nodeName());
}

void ZenoNode::onCollaspeBtnClicked()
{
    collaspe(!m_bCollasped);
    m_bCollasped = !m_bCollasped;
}

void ZenoNode::collaspe(bool collasped)
{
    if (collasped)
    {
        m_headerWidget->hide();
        m_bodyWidget->hide();
        for (auto p : m_inSocks) {
            p.second->hide();
        }
        for (auto p : m_outSocks) {
            p.second->hide();
        }
        m_mute->hide();
        m_view->hide();
        m_prep->hide();

        m_collaspedWidget->show();
        m_collaspe->toggle(true);
    }
    else
    {
        m_bodyWidget->show();
        for (auto p : m_inSocks) {
            p.second->show();
        }
        for (auto p : m_outSocks) {
            p.second->show();
        }
        m_mute->show();
        m_view->show();
        m_prep->show();
        m_headerWidget->show();
        m_collaspedWidget->hide();
        m_collaspe->toggle(false);
    }
    update();
}

QVariant ZenoNode::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        bool bSelected = isSelected();
        m_headerWidget->toggle(bSelected);
        m_bodyWidget->toggle(bSelected);
        m_collaspedWidget->toggle(bSelected);
    }
    else if (change == QGraphicsItem::ItemPositionChange)
    {
        emit nodePositionChange(nodeId());
    }
    else if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        QPointF pos = this->scenePos();
        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
        if (SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel))
        {
            pGraphModel->setData(m_index, pos, ROLE_OBJPOS);
        }
    }
    return value;
}