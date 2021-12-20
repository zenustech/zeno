#include "zenonode.h"
#include "../model/modelrole.h"
#include "../model/subgraphmodel.h"
#include "../render/common_id.h"
#include "zenoparamnameitem.h"
#include "zenoparamwidget.h"


ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_paramsWidget(nullptr)
    , m_headerBg(nullptr)
    , m_bodyBg(nullptr)
    , m_bInitSockets(false)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
{
    setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges | ItemSendsGeometryChanges);
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

    //better to calculate the width by layout item without manually calculate it.
    int width = m_renderParams.headerBg.rc.width();
    const QString &name = m_index.data(ROLE_OBJNAME).toString();
    QFontMetrics metrics(m_renderParams.paramFont);
    width = std::max(width, m_renderParams.namePos.toPoint().x() + metrics.horizontalAdvance(name) + m_renderParams.namePos.toPoint().x());
    m_renderParams.headerBg.rc.setWidth(width);
    m_renderParams.bodyBg.rc.setWidth(width);

    m_headerWidget = initHeaderBgWidget();
    initIndependentWidgets();
    m_bodyWidget = initBodyWidget();

    QGraphicsLinearLayout* pMainLayout = new QGraphicsLinearLayout(Qt::Vertical);
    pMainLayout->addItem(m_headerWidget);
    pMainLayout->addItem(m_bodyWidget);
    pMainLayout->setContentsMargins(0, 0, 0, 0);
    pMainLayout->setSpacing(0);

    setLayout(pMainLayout);

    //todo: border

    QPointF pos = m_index.data(ROLE_OBJPOS).toPointF();
    const QString &id = m_index.data(ROLE_OBJID).toString();
    setPos(pos);
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
}

ZenoBackgroundWidget* ZenoNode::initHeaderBgWidget()
{
    ZenoBackgroundWidget *headerWidget = new ZenoBackgroundWidget(m_renderParams.headerBg, this);
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

    pHLayout->addItem(new SpacerLayoutItem(QSizeF(horizontalPadding, 6), true));
    pHLayout->addItem(collaspeItem);
    pHLayout->addItem(new SpacerLayoutItem(QSizeF(horizontalPadding, 3), true));
    pHLayout->addItem(pNameItem);
    pHLayout->addItem(new SpacerLayoutItem(QSizeF(40, 0), true));

    pHLayout->setContentsMargins(0, 5, 0, 5);
    pHLayout->setSpacing(0);

    headerWidget->setLayout(pHLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);
    return headerWidget;
}

ZenoBackgroundWidget* ZenoNode::initBodyWidget()
{
    ZenoBackgroundWidget *bodyWidget = new ZenoBackgroundWidget(m_renderParams.bodyBg, this);
    QGraphicsLinearLayout *pVLayout = new QGraphicsLinearLayout(Qt::Vertical);

    if (QGraphicsGridLayout *pParamsLayout = initParams())
    {
        pParamsLayout->setContentsMargins(m_renderParams.distParam.paramsLPadding, 10, 10, 0);
        pVLayout->addItem(pParamsLayout);
    }
    if (QGraphicsGridLayout *pSocketsLayout = initSockets())
    {
        pSocketsLayout->setContentsMargins(m_renderParams.distParam.paramsLPadding, m_renderParams.distParam.paramsToTopSocket, m_renderParams.distParam.paramsLPadding, 0);
        pVLayout->addItem(pSocketsLayout);
    }

    bodyWidget->setLayout(pVLayout);
    bodyWidget->setZValue(ZVALUE_ELEMENT);
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
                    pParamsLayout->addItem(pMultiStrEdit, r, 1);
                    break;
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

QPointF ZenoNode::getPortPos(bool bInput, const QString &portName)
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
    if (event->type() == QEvent::MouseButtonPress)
    {
        int j;
        j = 0;
    }
    return _base::sceneEventFilter(watched, event);
}

bool ZenoNode::sceneEvent(QEvent *event)
{
    return _base::sceneEvent(event);
}

QVariant ZenoNode::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        bool bSelected = isSelected();
        if (m_headerBg)
            m_headerBg->toggle(bSelected);
        if (m_headerWidget)
            m_headerWidget->toggle(bSelected);
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