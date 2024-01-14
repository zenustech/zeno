#include "gvtestwidget.h"
#include "nodeeditor/gv/zgraphicslayoutitem.h"
#include "nodeeditor/gv/zgraphicslayoutitem.h"
#include "nodeeditor/gv/zlayoutbackground.h"
#include "nodeeditor/gv/zgraphicslayout.h"
#include "nodeeditor/gv/zsocketlayout.h"
#include "nodeeditor/gv/zenoparamwidget.h"
#include "nodeeditor/gv/nodesys_common.h"
#include "util/ztfutil.h"
#include "control/renderparam.h"


ZSocketLayout* socket4 = nullptr;
ZSocketLayout* socket1 = nullptr;
ZGraphicsLayout* pBodyLayout = nullptr;


TestGraphicsView::TestGraphicsView(QWidget* parent)
    : QGraphicsView(parent)
{
    setBackgroundBrush(QColor(24, 29, 33));

    QGraphicsScene* scene = new QGraphicsScene;

    ZLayoutBackground* pHeader = new ZLayoutBackground;
    pHeader->setColors(false, QColor(83, 96, 147), QColor(83, 96, 147), QColor(83, 96, 147));

    ZGraphicsLayout* pHeaderLayout = new ZGraphicsLayout(true);

    ZSimpleTextItem* pTitleItem = new ZSimpleTextItem("ExtractDict");
    pTitleItem->setBrush(QColor(226, 226, 226));
    QFont font2 = QApplication::font();
    font2.setPointSize(14);
    font2.setBold(true);
    pTitleItem->setFont(font2);
    pTitleItem->updateBoundingRect();

    pHeaderLayout->addSpacing(10);
    pHeaderLayout->addItem(pTitleItem, Qt::AlignVCenter);

    ZtfUtil& inst = ZtfUtil::GetInstance();
    NodeUtilParam m_nodeParams = inst.toUtilParam(inst.loadZtf(":/templates/node-example.xml"));
    ZenoMinStatusBtnWidget* pStatusWidgets = new ZenoMinStatusBtnWidget(m_nodeParams.status);

    pHeaderLayout->addSpacing(100);

    pHeaderLayout->addItem(pStatusWidgets);
    pHeader->setLayout(pHeaderLayout);


    pBodyLayout = new ZGraphicsLayout(false);
    pBodyLayout->setSpacing(5);
    pBodyLayout->setContentsMargin(16, 16, 16, 16);

    ZenoParamPushButton* spButton = new ZenoParamPushButton;
    spButton->setData(GVKEY_SIZEHINT, QSizeF(0, 32));
    spButton->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    spButton->setText("CLICK ME !!!");

    socket1 = new ZSocketLayout(QModelIndex(), true);
    socket1->setControl(spButton);

    ZSocketLayout *socket2 = new ZSocketLayout(QModelIndex(), true);
    socket4 = new ZSocketLayout(QModelIndex(), true);
    ZSocketLayout *socket3 = new ZSocketLayout(QModelIndex(), false);

    pBodyLayout->addLayout(socket1);
    pBodyLayout->addLayout(socket2);
    pBodyLayout->addLayout(socket4);
    pBodyLayout->addLayout(socket3);


    ZenoParamComboBox* spCombobox = new ZenoParamComboBox;
    spCombobox->setItems({ "SVGSD", "GVSF", "RGEGRE" });
    spCombobox->setData(GVKEY_SIZEHINT, QSizeF(0, 32));
    spCombobox->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    pBodyLayout->addItem(spCombobox);

    ZLayoutBackground* pBody = new ZLayoutBackground;
    pBody->setColors(false, QColor(30, 30, 30), QColor(30, 30, 30), QColor(30, 30, 30));
    pBody->setBorder(3, QColor(74, 72, 72));
    pBody->setLayout(pBodyLayout);

    ZGraphicsLayout* mainLayout = new ZGraphicsLayout(false);

    mainLayout->addItem(pHeader);
    mainLayout->addItem(pBody);
    mainLayout->setSpacing(0);

    ZLayoutBackground* pNode = new ZLayoutBackground;
    //pNode->setColors(false, QColor(255, 0, 0), QColor(255, 0, 0), QColor(255, 0, 0));
    pNode->setLayout(mainLayout);
    pNode->setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemIsSelectable);

    ZGraphicsLayout::updateHierarchy(mainLayout);

    scene->addItem(pNode);

    setScene(scene);
}