#include "zenonode.h"
#include "../model/modelrole.h"
#include "../model/subgraphmodel.h"
#include "../render/common_id.h"


ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
{
    setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges | ItemSendsGeometryChanges);
}

ZenoNode::~ZenoNode()
{
}

void ZenoNode::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    return;
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

    m_nameItem = new QGraphicsTextItem(this);
    const QString& name = m_index.data(ROLE_OBJNAME).toString();
    m_nameItem->setPlainText(name);
    m_nameItem->setPos(m_renderParams.namePos);
    m_nameItem->setZValue(ZVALUE_ELEMENT);
    m_nameItem->setFont(m_renderParams.nameFont);
    m_nameItem->setDefaultTextColor(m_renderParams.nameClr.color());

    QRectF rc;

    m_headerBg = new ZenoBackgroundItem(m_renderParams.headerBg, this);
    m_headerBg->setPos(m_renderParams.headerBg.rc.topLeft());
    m_headerBg->setZValue(ZVALUE_BACKGROUND);

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

    m_bodyBg = new ZenoBackgroundItem(m_renderParams.bodyBg, this);
    m_bodyBg->setPos(m_renderParams.bodyBg.rc.topLeft());
    m_bodyBg->setZValue(ZVALUE_ELEMENT);

    int x = m_bodyBg->pos().x(), y = m_bodyBg->pos().y();

    y += m_renderParams.distParam.paramsVPadding;

    int width = m_renderParams.headerBg.rc.width();

    initParams(y, width);
    initSockets(y, width);

    //needs to adjust body background height
    rc = m_renderParams.bodyBg.rc;
    m_headerBg->resize(QSizeF(width, m_renderParams.headerBg.rc.height()));
    m_bodyBg->resize(QSizeF(width, y - m_renderParams.headerBg.rc.height()));

    //todo: border
}

void ZenoNode::initParams(int &y, int& width)
{
    int x = m_bodyBg->pos().x() + m_renderParams.distParam.paramsLPadding;

    const QJsonObject &params = m_index.data(ROLE_PARAMETERS).toJsonObject();
    for (auto key : params.keys())
    {
        QJsonValue val = params.value(key);
        QString value;
        if (val.isString())
        {
            value = val.toString();
        }
        else if (val.isDouble())
        {
            //todo
            value = QString::number(val.toDouble());
        }
        //temp text item to show parameter.
        QString showText = key + ":\t" + value;
        QGraphicsTextItem *pParamItem = new QGraphicsTextItem(showText, this);
        pParamItem->setPos(x, y);
        pParamItem->setZValue(ZVALUE_ELEMENT);
        pParamItem->setFont(m_renderParams.paramFont);
        pParamItem->setDefaultTextColor(m_renderParams.paramClr.color());

        QFontMetrics metrics(m_renderParams.paramFont);
        width = std::max(width, x + metrics.horizontalAdvance(showText));

        y += m_renderParams.distParam.paramsVSpacing;
    }
    y += m_renderParams.distParam.paramsBottomPadding;
    if (!params.keys().isEmpty())
        y += m_renderParams.distParam.paramsToTopSocket;
}

void ZenoNode::initSockets(int& y, int& width)
{
    int x = m_bodyBg->pos().x() - m_renderParams.socketHOffset;
    const QJsonObject &inputs = m_index.data(ROLE_INPUTS).toJsonObject();
    for (auto key : inputs.keys())
    {
        const QString &name = key;
        ZenoImageItem *socket = new ZenoImageItem(m_renderParams.socket, m_renderParams.szSocket, this);
        QGraphicsTextItem *socketName = new QGraphicsTextItem(name, this);
        m_inSocks.insert(std::make_pair(name, socket));
        socket->setPos(QPointF(x, y));
        socket->setZValue(ZVALUE_ELEMENT);

        int x_sockettext = x + m_renderParams.socketToText + m_renderParams.szSocket.width();
        static const int textYOffset = 5;//text offset exists
        socketName->setPos(QPointF(x_sockettext, y - textYOffset));
        socketName->setZValue(ZVALUE_ELEMENT);
        socketName->setFont(m_renderParams.socketFont);
        socketName->setDefaultTextColor(m_renderParams.socketClr.color());
        y += m_renderParams.szSocket.height() + m_renderParams.socketVMargin;
    }

    x = m_bodyBg->pos().x() + m_renderParams.bodyBg.rc.width() - m_renderParams.socketHOffset;
    const QJsonObject &outputs = m_index.data(ROLE_OUTPUTS).toJsonObject();
    for (auto outputPort : outputs.keys())
    {
        ZenoImageItem *socket = new ZenoImageItem(m_renderParams.socket, m_renderParams.szSocket, this);
        QGraphicsTextItem *socketName = new QGraphicsTextItem(outputPort, this);
        m_outSocks.insert(std::make_pair(outputPort, socket));

        socket->setPos(QPointF(x, y));
        socket->setZValue(ZVALUE_ELEMENT);

        socketName->setZValue(ZVALUE_ELEMENT);
        socketName->setFont(m_renderParams.socketFont);
        socketName->setDefaultTextColor(m_renderParams.socketClr.color());
        QFontMetrics fontMetrics(m_renderParams.socketFont);
        int textWidth = fontMetrics.horizontalAdvance(outputPort);
        int x_sockettext = x - m_renderParams.socketToText - textWidth;
        static const int textYOffset = 5;
        socketName->setPos(QPointF(x_sockettext, y - textYOffset));

        y += m_renderParams.szSocket.height() + m_renderParams.socketVMargin;
    }
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

QJsonObject ZenoNode::inputParams() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_INPUTS).toJsonObject();
}

QJsonObject ZenoNode::outputParams() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OUTPUTS).toJsonObject();
}

QVariant ZenoNode::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        bool bSelected = isSelected();
        m_headerBg->toggle(bSelected);
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