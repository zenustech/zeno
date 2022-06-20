#include "makedictnode.h"
#include <zenoui/render/common_id.h>
#include <zenoui/style/zenostyle.h>


MakeDictNode::MakeDictNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
}

MakeDictNode::~MakeDictNode()
{
}

QGraphicsLayout* MakeDictNode::initParams()
{
    return ZenoNode::initParams();
}

QGraphicsLinearLayout* MakeDictNode::initCustomParamWidgets()
{
    //dict input control
    const QString &paramName = "dict";
    QGraphicsLinearLayout *pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);
    ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem(paramName, m_renderParams.paramFont, m_renderParams.paramClr.color());
    pNameItem->setTextInteractionFlags(Qt::TextBrowserInteraction);
    pHLayout->addItem(pNameItem);

    //ZenoParamPushButton *pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    //pHLayout->addItem(pEditBtn);

    ImageElement elem;
    elem.image = ":/icons/toggle-off.svg";
    elem.imageOn = ":/icons/toggle-on.svg";

    ZenoSvgLayoutItem *pToggleBtn = new ZenoSvgLayoutItem(elem, ZenoStyle::dpiScaledSize(QSize(56, 30)));
    pHLayout->addItem(pToggleBtn);
    pToggleBtn->setCheckable(true);

    pHLayout->addStretch();
    //connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    //ZenoSocketItem *socket = new ZenoSocketItem(m_renderParams.socket, m_renderParams.szSocket, this);
    //socket->setZValue(ZVALUE_ELEMENT);
    //registerParamSocket(paramName, socket, pNameItem, pEditBtn);
    return pHLayout;
}