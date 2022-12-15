#include "zdictpanel.h"
#include "zsocketlayout.h"
#include "../../render/renderparam.h"
#include "../../style/zenostyle.h"
#include "zenosocketitem.h"


class ZDictItemLayout : public ZGraphicsLayout
{
public:
    ZDictItemLayout() : ZGraphicsLayout(true)
    {
        //setContentsMargin(0, 10, 0, 10);

        ImageElement elem;
        elem.image = ":/icons/socket-off.svg";
        elem.imageHovered = ":/icons/socket-hover.svg";
        elem.imageOn = ":/icons/socket-on.svg";
        elem.imageOnHovered = ":/icons/socket-on-hover.svg";

        const int cSocketWidth = 10;
        const int cSocketHeight = 10;

        m_socket = new ZenoSocketItem(QModelIndex(), true, elem, ZenoStyle::dpiScaledSize(QSizeF(cSocketWidth, cSocketHeight)));
        qreal leftMargin = ZenoStyle::dpiScaled(10);
        qreal rightMargin = ZenoStyle::dpiScaled(10);
        qreal topMargin = ZenoStyle::dpiScaled(10);
        qreal bottomMargin = ZenoStyle::dpiScaled(10);
        m_socket->setContentMargins(leftMargin, topMargin, rightMargin, bottomMargin);

        QObject::connect(m_socket, &ZenoSocketItem::clicked, [=]() {
            int j;
            j = 0;
        });

        m_editText = new ZGraphicsLayoutItem<ZGraphicsTextItem>();
        {
            m_editText->setDefaultTextColor(QColor(188, 188, 188));
            QFont font("HarmonyOS Sans Bold", 11);
            font.setBold(true);
            m_editText->setFont(font);

            m_editText->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
            m_editText->setData(GVKEY_SIZEHINT, ZenoStyle::dpiScaledSize(QSizeF(128, 32)));
            m_editText->setTextInteractionFlags(Qt::TextEditorInteraction);

            QTextFrame* frame = m_editText->document()->rootFrame();
            QTextFrameFormat format = frame->frameFormat();
            format.setBackground(QColor(37, 37, 37));
            frame->setFrameFormat(format);

            m_editText->setFlag(QGraphicsItem::ItemSendsGeometryChanges);
            m_editText->setFlag(QGraphicsItem::ItemSendsScenePositionChanges);
        }

        addItem(m_socket, Qt::AlignVCenter);
        addItem(m_editText);
        addSpacing(0, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred));
    }

private:
    ZenoSocketItem* m_socket;
    ZGraphicsLayoutItem<ZGraphicsTextItem>* m_editText;
};




ZDictPanel::ZDictPanel()
    : ZLayoutBackground()
{
    setRadius(10, 10, 10, 10);
    setColors(false, QColor(50, 50, 50), QColor(50, 50, 50), QColor(50, 50, 50));
    setBorder(0, QColor());

    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(false);

    pVLayout->setContentsMargin(8, 12, 8, 12);
    pVLayout->setSpacing(8);

    ZDictItemLayout* pkey = new ZDictItemLayout;
    pkey->setDebugName("debugdict");
    pVLayout->addLayout(pkey);
    ZDictItemLayout* pkey2 = new ZDictItemLayout;
    pVLayout->addLayout(pkey2);

    setLayout(pVLayout);
}