#include "framework.h"
#include "nodetemplate.h"
#include "nodescene.h"
#include "resizableitemimpl.h"
#include "resizecoreitem.h"
#include "styletabwidget.h"


class RAII_BLOCKSIGNAL
{
public:
    RAII_BLOCKSIGNAL(QObject *pObject) : m_pObject(pObject) {
        if (m_pObject)
            m_pObject->blockSignals(true);
    }
    ~RAII_BLOCKSIGNAL() {
        if (m_pObject)
            m_pObject->blockSignals(false);
    }

private:
    QObject *m_pObject;
};

class RAII_BATCHLEVEL {
public:
    RAII_BATCHLEVEL(int *level) : m_level(level) {
        if (m_level)
            (*m_level)++;
    }
    ~RAII_BATCHLEVEL() {
        if (m_level)
            (*m_level)--;
    }

private:
    int *m_level;
};



NodeTemplate::NodeTemplate(NodeScene* pScene, QGraphicsItem* parent)
    : QGraphicsObject(parent)
    , m_pScene(pScene)
	, m_once(nullptr)
	, m_prep(nullptr)
	, m_mute(nullptr)
	, m_view(nullptr)
	, m_genshin(nullptr)
	, m_background(nullptr)
	, m_nodename(nullptr)
	, m_component_nodename(nullptr)
	, m_component_status(nullptr)
	, m_component_control(nullptr)
	, m_component_display(nullptr)
	, m_component_header_backboard(nullptr)
	, m_component_ltsocket(nullptr)
    , m_component_body_backboard(nullptr)
	, m_model(new QStandardItemModel(pScene->views()[0]))
	, m_selection(new QItemSelectionModel(m_model))
    , m_batchLevel(0)
{
    m_model->setObjectName(NODE_MODEL_NAME);
    m_selection->setObjectName(NODE_SELECTION_MODEL);
	pScene->addItem(this);
}

QStandardItem* NodeTemplate::initModelItemFromGvItem(ResizableItemImpl *gvItem, const QString &id, const QString &name)
{
    QStandardItem *pItem = new QStandardItem(QIcon(), name);
    
	pItem->setData(id, NODEID_ROLE);
    pItem->setData(gvItem->coreItemSceneRect(), NODEPOS_ROLE);
    pItem->setData(gvItem->isLocked(), NODELOCK_ROLE);
    pItem->setData(gvItem->isVisible(), NODEVISIBLE_ROLE);
    pItem->setData(gvItem->getType(), NODETYPE_ROLE);
    pItem->setData(gvItem->getContent(), NODECONTENT_ROLE);
    pItem->setEditable(false);
	
	connect(gvItem, SIGNAL(gvItemSelectedChange(QString, bool)), this, SLOT(onGvItemSelectedChange(QString, bool)));
    connect(gvItem, SIGNAL(gvItemGeoChanged(QString, QRectF)), this, SLOT(onGvItemGeoChanged(QString, QRectF)));

    return pItem;
}

void NodeTemplate::setComponentPxElem(QStandardItem *pParentItem, ResizableItemImpl *pComponentObj, const ImageElement &imgElem, const QString &showName)
{
    pComponentObj->setContent(NC_IMAGE);
    QString id = pComponentObj->getId();
    QStandardItem *pItem = initModelItemFromGvItem(pComponentObj, id, showName );
    pParentItem->appendRow(pItem);
    pItem->setData(imgElem.image, NODEPATH_ROLE);
    pItem->setData(imgElem.imageHovered, NODEHOVERPATH_ROLE);
    pItem->setData(imgElem.imageOn, NODESELECTEDPATH_ROLE);
    QString path = m_param.header.backboard.imageElem.image;
    QSizeF sz(imgElem.rc.width(), imgElem.rc.height());
    ResizableImageItem *pPxItem = new ResizableImageItem(imgElem.image, imgElem.imageHovered, imgElem.imageOn, sz);
    pComponentObj->setCoreItem(pPxItem);
}

void NodeTemplate::setupBackground(QStandardItem* pParentItem, ResizableItemImpl* pComponentObj, const BackgroundComponent& bg, const QString& showName)
{
    pComponentObj->setContent(NC_BACKGROUND);
    QString id = pComponentObj->getId();
    QStandardItem *pItem = initModelItemFromGvItem(pComponentObj, id, showName);
    pParentItem->appendRow(pItem);

    pItem->setData(bg.clr_normal, NODECOLOR_NORMAL_ROLE);
    pItem->setData(bg.clr_hovered, NODECOLOR_HOVERD_ROLE);
    pItem->setData(bg.clr_selected, NODECOLOR_SELECTED_ROLE);
    pItem->setData(bg.lt_radius, NODE_LTRADIUS_ROLE);
    pItem->setData(bg.rt_radius, NODE_RTRADIUS_ROLE);
    pItem->setData(bg.lb_radius, NODE_LBRADIUS_ROLE);
    pItem->setData(bg.rb_radius, NODE_RBRADIUS_ROLE);
    pItem->setData(bg.imageElem.image, NODEPATH_ROLE);
    pItem->setData(bg.imageElem.imageHovered, NODEHOVERPATH_ROLE);
    pItem->setData(bg.imageElem.imageOn, NODESELECTEDPATH_ROLE);

    QSizeF sz(bg.rc.width(), bg.rc.height());
    ResizableRectItem *pRcItem = new ResizableRectItem(bg);
    pRcItem->setColors(bg.clr_normal, bg.clr_hovered, bg.clr_selected);
    pComponentObj->setCoreItem(pRcItem);
}

void NodeTemplate::setComonentTxtElem(QStandardItem* pParentItem, ResizableItemImpl* pComponentObj, const TextElement& textElem, const QString& showName)
{
    pComponentObj->setContent(NC_TEXT);
    QString id = pComponentObj->getId();
    QStandardItem *pItem = initModelItemFromGvItem(pComponentObj, id, showName);
    pParentItem->appendRow(pItem);
    pItem->setData(textElem.font, NODEFONT_ROLE);
    pItem->setData(textElem.fill, NODEFONTCOLOR_ROLE);
    pItem->setData(textElem.text, NODETEXT_ROLE);

    ResizableTextItem *pNameItem = new ResizableTextItem(textElem.text);
    pNameItem->setTextProp(textElem.font, textElem.fill.color());
    pComponentObj->setCoreItem(pNameItem);
}

void NodeTemplate::addImageElement(QStandardItem* pParentItem, const ImageElement& imgElem, ResizableItemImpl* pComponentObj, const QString& showName)
{
    const QString& id = imgElem.id;
    auto elem = new ResizableItemImpl(NT_ELEMENT, id, imgElem.rc, pComponentObj);
    elem->setContent(NC_IMAGE);
    m_objs.insert(std::make_pair(id, elem));
    if (!imgElem.image.isEmpty())
    {
        QSizeF sz(imgElem.rc.width(), imgElem.rc.height());
        auto pxItem = new ResizableImageItem(imgElem.image, imgElem.imageHovered, imgElem.imageOn, sz);
        elem->setCoreItem(pxItem);
    }
    QStandardItem* pItem = initModelItemFromGvItem(elem, id, showName);
    pItem->setData(imgElem.image, NODEPATH_ROLE);
    pItem->setData(imgElem.imageHovered, NODEHOVERPATH_ROLE);
    pItem->setData(imgElem.imageOn, NODESELECTEDPATH_ROLE);
    pParentItem->appendRow(pItem);
}

void NodeTemplate::addTextElement(QStandardItem *pParentItem, const TextElement &textElem, ResizableItemImpl *pComponentObj, const QString &showName)
{
    const QString &id = textElem.id;
    auto elem = new ResizableItemImpl(NT_ELEMENT, id, textElem.rc, pComponentObj);
    elem->setContent(NC_TEXT);
    m_objs.insert(std::make_pair(id, elem));
    if (!textElem.text.isEmpty())
    {
        elem->setCoreItem(new ResizableTextItem(textElem.text));
    }
    QStandardItem *pItem = initModelItemFromGvItem(elem, id, showName);
    pItem->setData(textElem.text, NODETEXT_ROLE);
    pParentItem->appendRow(pItem);
}

void NodeTemplate::initStyleModel(const NodeParam& param)
{
	m_param = param;

	m_model->clear();

    QStandardItem *headerItem = new QStandardItem(QIcon(), "Header");
    headerItem->setData(HEADER_ID);
    headerItem->setEditable(false);
    headerItem->setSelectable(false);

    QString id = m_param.header.name.id;
    //move as element, actually a component-element
    m_component_nodename = new ResizableItemImpl(NT_ELEMENT, id, m_param.header.name.rc, this);
    setComonentTxtElem(headerItem, m_component_nodename, m_param.header.name.text, "Node-name");
    m_objs.insert(std::make_pair(id, m_component_nodename));

    id = m_param.header.status.id;
    m_component_status = new ResizableItemImpl(NT_COMPONENT, id, m_param.header.status.rc, this);
    m_objs.insert(std::make_pair(id, m_component_status));
    QStandardItem *pStatusItem = initModelItemFromGvItem(m_component_status, id, "Status");
    headerItem->appendRow(pStatusItem);
    {
        addImageElement(pStatusItem, m_param.header.status.mute, m_component_status, "Mute");
        addImageElement(pStatusItem, m_param.header.status.view, m_component_status, "View");
        addImageElement(pStatusItem, m_param.header.status.prep, m_component_status, "Prep");
    }

    id = m_param.header.backboard.id;
    m_component_header_backboard = new ResizableItemImpl(NT_COMPONENT_AS_ELEMENT, id, m_param.header.backboard.rc, this);
    setupBackground(headerItem, m_component_header_backboard, m_param.header.backboard, "Back-board");
    m_objs.insert(std::make_pair(id, m_component_header_backboard));

    id = m_param.header.display.id;
    m_component_display = new ResizableItemImpl(NT_COMPONENT_AS_ELEMENT, id, m_param.header.display.rc, this);
    setComponentPxElem(headerItem, m_component_display, m_param.header.display.image, "Display");
    m_objs.insert(std::make_pair(id, m_component_display));

    id = m_param.header.control.id;
    m_component_control = new ResizableItemImpl(NT_COMPONENT, id, m_param.header.control.rc, this);
    m_objs.insert(std::make_pair(id, m_component_control));
    QStandardItem *controlItem = initModelItemFromGvItem(m_component_control, id, "Control");
    headerItem->appendRow(controlItem);
    {
        const QVector<ImageElement>& elems = m_param.header.control.elements;
        for (int i = 0; i < elems.size(); i++)
        {
            const ImageElement& elem = elems[i];
            addImageElement(controlItem, elem, m_component_control, "Collaspe.svg");
        }
    }

	QStandardItem *bodyItem = new QStandardItem(QIcon(), "Body");
    bodyItem->setData(BODY_ID);
    bodyItem->setEditable(false);
    bodyItem->setSelectable(false);

    id = m_param.body.leftTopSocket.id;
    m_component_ltsocket = new ResizableItemImpl(NT_COMPONENT, id, m_param.body.leftTopSocket.rc, this);
    m_objs.insert(std::make_pair(id, m_component_ltsocket));
    QStandardItem *pLTSocketItem = initModelItemFromGvItem(m_component_ltsocket, id, "LTSocket");
    bodyItem->appendRow(pLTSocketItem);
    {
        addTextElement(pLTSocketItem, m_param.body.leftTopSocket.text, m_component_ltsocket, "socket text");
        addImageElement(pLTSocketItem, m_param.body.leftTopSocket.image, m_component_ltsocket, "socket image");
    }

    id = m_param.body.leftBottomSocket.id;
    m_component_lbsocket = new ResizableItemImpl(NT_COMPONENT, id, m_param.body.leftBottomSocket.rc, this);
    m_objs.insert(std::make_pair(id, m_component_lbsocket));
    QStandardItem *pLBSocketItem = initModelItemFromGvItem(m_component_lbsocket, id, "LBSocket");
    bodyItem->appendRow(pLBSocketItem);
    {
        addTextElement(pLBSocketItem, m_param.body.leftBottomSocket.text, m_component_lbsocket, "socket text");
        addImageElement(pLBSocketItem, m_param.body.leftBottomSocket.image, m_component_lbsocket, "socket image");
    }

    id = m_param.body.rightTopSocket.id;
    m_component_rtsocket = new ResizableItemImpl(NT_COMPONENT, id, m_param.body.rightTopSocket.rc, this);
    m_objs.insert(std::make_pair(id, m_component_rtsocket));
    QStandardItem *pRTSocketItem = initModelItemFromGvItem(m_component_rtsocket, id, "RTSocket");
    bodyItem->appendRow(pRTSocketItem);
    {
        addTextElement(pRTSocketItem, m_param.body.rightTopSocket.text, m_component_rtsocket, "socket text");
        addImageElement(pRTSocketItem, m_param.body.rightTopSocket.image, m_component_rtsocket, "socket image");
    }

    id = m_param.body.rightBottomSocket.id;
    m_component_rbsocket = new ResizableItemImpl(NT_COMPONENT, id, m_param.body.rightBottomSocket.rc, this);
    m_objs.insert(std::make_pair(id, m_component_rbsocket));
    QStandardItem *pRBSocketItem = initModelItemFromGvItem(m_component_rbsocket, id, "RBSocket");
    bodyItem->appendRow(pRBSocketItem);
    {
        addTextElement(pRBSocketItem, m_param.body.rightBottomSocket.text, m_component_rbsocket, "socket text");
        addImageElement(pRBSocketItem, m_param.body.rightBottomSocket.image, m_component_rbsocket, "socket image");
    }

    id = m_param.body.backboard.id;
    m_component_body_backboard = new ResizableItemImpl(NT_COMPONENT_AS_ELEMENT, id, QRectF(m_param.body.backboard.rc), this);
    setupBackground(bodyItem, m_component_body_backboard, m_param.body.backboard, "Back-board");
    m_objs.insert(std::make_pair(id, m_component_body_backboard));

    RAII_BLOCKSIGNAL batch(m_model);
	m_model->appendRow(headerItem);
	m_model->appendRow(bodyItem);

	connect(m_selection, SIGNAL(selectionChanged(const QItemSelection &, const QItemSelection &)),
        this, SLOT(onSelectionChanged(const QItemSelection &, const QItemSelection &)));
    connect(m_model, SIGNAL(itemChanged(QStandardItem *)), this, SLOT(onItemChanged(QStandardItem *)));
    connect(m_model, SIGNAL(dataChanged(const QModelIndex&, const QModelIndex&, const QVector<int>)),
            this, SLOT(onDataChanged(const QModelIndex &, const QModelIndex &, const QVector<int>)));
}

NodeParam NodeTemplate::exportParam()
{
    NodeParam param;
    param.header = _exportHeaderParam();
    param.body = _exportBodyParam();
    return param;
}

BodyParam NodeTemplate::_exportBodyParam()
{
    BodyParam param;
    param.leftTopSocket = _exportSocket(COMPONENT_LTSOCKET);
    param.leftBottomSocket = _exportSocket(COMPONENT_LBSOCKET);
    param.rightTopSocket = _exportSocket(COMPONENT_RTSOCKET);
    param.rightBottomSocket = _exportSocket(COMPONENT_RBSOCKET);
    param.backboard = _exportBackgroundComponent(COMPONENT_BODY_BG);
    return param;
}
    
HeaderParam NodeTemplate::_exportHeaderParam()
{
    HeaderParam param;
    param.status = _exportStatusComponent(COMPONENT_STATUS);
    param.control = _exportControlComponent(COMPONENT_CONTROL);
    param.display = _exportImageComponent(COMPONENT_DISPLAY);
    param.name = _exportNameComponent(COMPONENT_NAME);
    param.backboard = _exportBackgroundComponent(COMPONENT_HEADER_BG);
    return param;
}

TextComponent NodeTemplate::_exportNameComponent(QString id)
{
    TextComponent comp;
    QStandardItem *pItem = getItemFromId(id);
    comp.id = id;
    comp.rc = pItem->data(NODEPOS_ROLE).toRect();

    TextElement elem;
    elem.text = pItem->data(NODETEXT_ROLE).toString();
    elem.font = qvariant_cast<QFont>(pItem->data(NODEFONT_ROLE));
    elem.fill = qvariant_cast<QBrush>(pItem->data(NODEFONTCOLOR_ROLE)); //QColor 2 QBrush
    elem.rc = comp.rc;

    comp.text = elem;
    return comp;
}

SocketComponent NodeTemplate::_exportSocket(QString id)
{
    SocketComponent comp;
    QStandardItem* pItem = getItemFromId(id);
    comp.id = id;
    comp.rc = pItem->data(NODEPOS_ROLE).toRect();
    if (id == COMPONENT_LTSOCKET) {
        comp.image = _exportImageElement(ELEMENT_LTSOCKET_IMAGE);
        comp.text = _exportTextElement(ELEMENT_LTSOCKET_TEXT);
    }
    else if (id == COMPONENT_LBSOCKET) {
        comp.image = _exportImageElement(ELEMENT_LBSOCKET_IMAGE);
        comp.text = _exportTextElement(ELEMENT_LBSOCKET_TEXT);
    }
    else if (id == COMPONENT_RTSOCKET) {
        comp.image = _exportImageElement(ELEMENT_RTSOCKET_IMAGE);
        comp.text = _exportTextElement(ELEMENT_RTSOCKET_TEXT);
    }
    else if (id == COMPONENT_RBSOCKET) {
        comp.image = _exportImageElement(ELEMENT_RBSOCKET_IMAGE);
        comp.text = _exportTextElement(ELEMENT_RBSOCKET_TEXT);
    }
    return comp;
}

ImageElement NodeTemplate::_exportImageElement(QString id) {
    ImageElement elem;
    QStandardItem *pItem = getItemFromId(id);
    elem.id = id;
    elem.rc = pItem->data(NODEPOS_ROLE).toRectF();
    elem.image = pItem->data(NODEPATH_ROLE).toString();
    elem.imageHovered = pItem->data(NODEHOVERPATH_ROLE).toString();
    elem.imageOn = pItem->data(NODESELECTEDPATH_ROLE).toString();
    return elem;
}

TextElement NodeTemplate::_exportTextElement(QString id) {
    TextElement elem;
    QStandardItem *pItem = getItemFromId(id);
    elem.id = id;
    elem.font = qvariant_cast<QFont>(pItem->data(NODEFONT_ROLE));
    elem.text = pItem->data(NODETEXT_ROLE).toString();
    elem.fill = qvariant_cast<QColor>(pItem->data(NODEFONTCOLOR_ROLE));
    QRectF rc = pItem->data(NODEPOS_ROLE).toRectF();
    elem.rc = rc;
    return elem;
}

BackgroundComponent NodeTemplate::_exportBackgroundComponent(QString id)
{
    BackgroundComponent comp;
    QStandardItem *pItem = getItemFromId(id);
    comp.id = id;
    comp.rc = pItem->data(NODEPOS_ROLE).toRect();

    ImageElement elem;
    elem.id = ELEMENT_BODY_BG;
    elem.rc = pItem->data(NODEPOS_ROLE).toRectF();
    elem.image = pItem->data(NODEPATH_ROLE).toString();
    elem.imageHovered = pItem->data(NODEHOVERPATH_ROLE).toString();
    elem.imageOn = pItem->data(NODESELECTEDPATH_ROLE).toString();

    comp.clr_normal = qvariant_cast<QColor>(pItem->data(NODECOLOR_NORMAL_ROLE));
    comp.clr_hovered = qvariant_cast<QColor>(pItem->data(NODECOLOR_HOVERD_ROLE));
    comp.clr_selected = qvariant_cast<QColor>(pItem->data(NODECOLOR_SELECTED_ROLE));

    comp.lt_radius = pItem->data(NODE_LTRADIUS_ROLE).toInt();
    comp.rt_radius = pItem->data(NODE_RTRADIUS_ROLE).toInt();
    comp.lb_radius = pItem->data(NODE_LBRADIUS_ROLE).toInt();
    comp.rb_radius = pItem->data(NODE_RBRADIUS_ROLE).toInt();

    return comp;
}

ImageComponent NodeTemplate::_exportImageComponent(QString id)
{
    ImageComponent comp;
    QStandardItem *pItem = getItemFromId(id);
    comp.id = id;
    comp.rc = pItem->data(NODEPOS_ROLE).toRect();

    ImageElement elem;
    if (id == COMPONENT_HEADER_BG)
    {
        elem.id = ELEMENT_HEADER_BG;
    }
    else if (id == COMPONENT_BODY_BG)
    {
        elem.id = ELEMENT_BODY_BG;
    }
    else if (id == COMPONENT_DISPLAY)
    {
        elem.id = ELEMENT_DISPLAY;
    }

    elem.rc = pItem->data(NODEPOS_ROLE).toRectF();
    elem.image = pItem->data(NODEPATH_ROLE).toString();
    elem.imageHovered = pItem->data(NODEHOVERPATH_ROLE).toString();
    elem.imageOn = pItem->data(NODESELECTEDPATH_ROLE).toString();
    comp.image = elem;

    return comp;
}

StatusComponent NodeTemplate::_exportStatusComponent(QString id)
{
    StatusComponent comp;
    QStandardItem *pItem = getItemFromId(id);
    comp.id = id;
    comp.rc = pItem->data(NODEPOS_ROLE).toRect();
    comp.mute = _exportImageElement(ELEMENT_MUTE);
    comp.view = _exportImageElement(ELEMENT_VIEW);
    comp.prep = _exportImageElement(ELEMENT_PREP);
    return comp;
}

Component NodeTemplate::_exportControlComponent(QString id)
{
    Component comp;
    QStandardItem *pItem = getItemFromId(id);
    comp.id = id;
    comp.rc = pItem->data(NODEPOS_ROLE).toRect();
    comp.elements.append(_exportImageElement(ELEMENT_COLLAPSE));
    return comp;
}

QVariant NodeTemplate::itemChange(GraphicsItemChange change, const QVariant& value)
{
	return QGraphicsObject::itemChange(change, value);
}

void NodeTemplate::onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected)
{
	QModelIndexList lst = selected.indexes();
	if (!lst.isEmpty())
	{
		QModelIndex idx = lst.at(0);
        QString id = idx.data(NODEID_ROLE).toString();
        if (m_objs.find(id) == m_objs.end())
            return;

        RAII_BLOCKSIGNAL batch(m_objs[id]);
        m_objs[id]->setSelected(true);
	}

	lst = deselected.indexes();
	if (!lst.isEmpty())
	{
		QModelIndex idx = lst.at(0);
        QString id = idx.data(NODEID_ROLE).toString();
        if (m_objs.find(id) == m_objs.end())
            return;

		RAII_BLOCKSIGNAL batch(m_objs[id]);
        m_objs[id]->setSelected(false);
	}
}

void NodeTemplate::onGvItemSelectedChange(QString id, bool selected)
{
    QStandardItem *pItemChanged = getItemFromId(id);
    QRectF rc = pItemChanged->data(NODEPOS_ROLE).toRectF();
	auto it = m_objs.find(id);
	if (it != m_objs.end())
	{
        m_selection->select(pItemChanged->index(), selected ? QItemSelectionModel::Select : QItemSelectionModel::Deselect);
	}
}

QStandardItem* NodeTemplate::getItemFromId(const QString &id)
{
    QModelIndexList lst = m_model->match(m_model->index(0, 0), NODEID_ROLE, id, 1, Qt::MatchRecursive);
    if (lst.isEmpty())
        return nullptr;
    Q_ASSERT(lst.size() == 1);
    return m_model->itemFromIndex(lst[0]);
}

void NodeTemplate::onGvItemGeoChanged(QString id, QRectF sceneRect)
{
    if (m_batchLevel > 0)
        return;

    RAII_BATCHLEVEL batch(&m_batchLevel);

    QStandardItem *pItemChanged = getItemFromId(id);
    pItemChanged->setData(sceneRect, NODEPOS_ROLE);

    //update children pos to model
    ResizableItemImpl* gvItem = m_objs.find(id)->second;
    QList<QGraphicsItem*> children = gvItem->childItems();
    foreach(QGraphicsItem* childItem, children)
    {
        //temp code: need a cast way for graphicsitem because qgraphicsitem_cast is only a static_cast but not "metacast".
        ResizableItemImpl* pChild = dynamic_cast<ResizableItemImpl*>(childItem);
        if (pChild) {
            QRectF rcChild = pChild->coreItemSceneRect();
            QStandardItem *pChildItemChanged = getItemFromId(pChild->getId());
            if (pChildItemChanged)
            {
                pChildItemChanged->setData(rcChild, NODEPOS_ROLE);
            }
        }
    }
}

void NodeTemplate::onItemChanged(QStandardItem *pItem)
{
    if (m_batchLevel > 0)
        return;

    RAII_BATCHLEVEL batch(&m_batchLevel);

    QString id = pItem->data(NODEID_ROLE).toString();
    auto it = m_objs.find(id);
    if (it != m_objs.end())
	{
        ResizableItemImpl *gvItem = it->second;
        QRectF rc = pItem->data(NODEPOS_ROLE).toRectF();
        gvItem->setCoreItemSceneRect(rc);
        bool bLock = pItem->data(NODELOCK_ROLE).toBool();
        bool bVisible = pItem->data(NODEVISIBLE_ROLE).toBool();
        gvItem->setVisible(bVisible);
        gvItem->setLocked(bLock);
    }
}

void NodeTemplate::onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int>& roles)
{
    if (roles.contains(NODEPATH_ROLE) || roles.contains(NODEHOVERPATH_ROLE) || roles.contains(NODESELECTEDPATH_ROLE))
    {
        QString id = topLeft.data(NODEID_ROLE).toString();
        auto it = m_objs.find(id);
        if (it != m_objs.end())
        {
            ResizableItemImpl *gvItem = it->second;
            if (gvItem->getContent() == NC_IMAGE)
            {
                QSizeF sz = QSizeF(gvItem->width(), gvItem->height());
                QString normal = topLeft.data(NODEPATH_ROLE).toString();
                QString hovered = topLeft.data(NODEHOVERPATH_ROLE).toString();
                QString selected = topLeft.data(NODESELECTEDPATH_ROLE).toString();
                ResizableCoreItem* pCoreItem = gvItem->coreItem();
                if (pCoreItem == nullptr) {
                    auto pixmapItem = new ResizableImageItem(normal, hovered, selected, sz);
                    gvItem->setCoreItem(pixmapItem);
                } else {
                    auto pixmapItem = qgraphicsitem_cast<ResizableImageItem *>(pCoreItem);
                    pixmapItem->resetImage(normal, hovered, selected, sz);
                }
            }
            if (gvItem->getContent() == NC_BACKGROUND)
            {
                QSizeF sz = QSizeF(gvItem->width(), gvItem->height());
                QString normal = topLeft.data(NODEPATH_ROLE).toString();
                QString hovered = topLeft.data(NODEHOVERPATH_ROLE).toString();
                QString selected = topLeft.data(NODESELECTEDPATH_ROLE).toString();
                ResizableCoreItem* pCoreItem = gvItem->coreItem();
                auto rectItem = qgraphicsitem_cast<ResizableRectItem *>(pCoreItem);
                //todo...
            }
        }
    } 
    else if (roles.contains(NODEFONT_ROLE) || roles.contains(NODEFONTCOLOR_ROLE))
    {
        QString id = topLeft.data(NODEID_ROLE).toString();
        auto it = m_objs.find(id);
        if (it != m_objs.end())
        {
            ResizableItemImpl *gvItem = it->second;
            if (gvItem->getContent() == NC_TEXT)
            {
                QFont font = qvariant_cast<QFont>(topLeft.data(NODEFONT_ROLE));
                QColor fontColor = qvariant_cast<QColor>(topLeft.data(NODEFONTCOLOR_ROLE));
                QString text = topLeft.data(NODETEXT_ROLE).toString();
                ResizableTextItem* pCoreItem = qgraphicsitem_cast<ResizableTextItem*>(gvItem->coreItem());
                if (pCoreItem == nullptr)
                {
                    pCoreItem = new ResizableTextItem(text);
                    gvItem->setCoreItem(pCoreItem);
                }
                pCoreItem->setTextProp(font, fontColor);
            }
         }
    }
    else if (roles.contains(NODETEXT_ROLE))
    {
        QString id = topLeft.data(NODEID_ROLE).toString();
        auto it = m_objs.find(id);
        if (it != m_objs.end())
        {
            ResizableItemImpl *gvItem = it->second;
            if (gvItem->getContent() == NC_TEXT)
            {
                QString text = topLeft.data(NODETEXT_ROLE).toString();
                ResizableTextItem *pCoreItem = qgraphicsitem_cast<ResizableTextItem *>(gvItem->coreItem());
                if (pCoreItem)
                {
                    pCoreItem->setText(text);
                }
            }
            else if (gvItem->getContent() == NC_BACKGROUND)
            {
                ResizableRectItem *pCoreItem = qgraphicsitem_cast<ResizableRectItem *>(gvItem->coreItem());
            }
        }
    }
    else if (roles.contains(NODECOLOR_NORMAL_ROLE) || roles.contains(NODECOLOR_HOVERD_ROLE) || roles.contains(NODECOLOR_SELECTED_ROLE)
        || roles.contains(NODE_LTRADIUS_ROLE) || roles.contains(NODE_RTRADIUS_ROLE) 
        || roles.contains(NODE_LBRADIUS_ROLE) || roles.contains(NODE_RBRADIUS_ROLE))
    {
        QString id = topLeft.data(NODEID_ROLE).toString();
        auto it = m_objs.find(id);
        if (it != m_objs.end()) {
            ResizableItemImpl *gvItem = it->second;
            int ltradius = topLeft.data(NODE_LTRADIUS_ROLE).toInt();
            int rtradius = topLeft.data(NODE_RTRADIUS_ROLE).toInt();
            int lbradius = topLeft.data(NODE_LBRADIUS_ROLE).toInt();
            int rbradius = topLeft.data(NODE_RBRADIUS_ROLE).toInt();
            QColor normal = qvariant_cast<QColor>(topLeft.data(NODECOLOR_NORMAL_ROLE));
            QColor hovered = qvariant_cast<QColor>(topLeft.data(NODECOLOR_HOVERD_ROLE));
            QColor selected = qvariant_cast<QColor>(topLeft.data(NODECOLOR_SELECTED_ROLE));
            ResizableRectItem *pCoreItem = qgraphicsitem_cast<ResizableRectItem *>(gvItem->coreItem());
            pCoreItem->setColors(normal, hovered, selected);
            pCoreItem->setRadius(ltradius, rtradius, lbradius, rbradius);
        }
    }
    emit markDirty();
}

void NodeTemplate::onRowsInserted(const QModelIndex &parent, int first, int last)
{
}

QRectF NodeTemplate::boundingRect() const
{
	QRectF wtf = this->childrenBoundingRect();
    QSizeF sz = m_pScene->getSceneSize();
    return QRectF(0, 0, sz.width(), sz.height());
}

QPainterPath NodeTemplate::shape() const
{
	return QGraphicsObject::shape();
}

QStandardItemModel *NodeTemplate::model() const
{
    return m_model;
}

QItemSelectionModel *NodeTemplate::selectionModel() const
{
    return m_selection;
}

void NodeTemplate::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}
