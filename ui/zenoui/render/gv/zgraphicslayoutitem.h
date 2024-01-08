#ifndef __ZGRAPHICS_LAYOUT_ITEM_H__
#define __ZGRAPHICS_LAYOUT_ITEM_H__

#include <QGraphicsObject>
#include <QSizePolicy>
#include "zgraphicslayout.h"


template <class T>
class ZGraphicsLayoutItem : public T
{
public:
    ZGraphicsLayoutItem(QGraphicsItem* parent = nullptr)
        : T(parent)
        , m_layout(nullptr)
        , m_parentLayout(nullptr)
    {
    }

    ~ZGraphicsLayoutItem()
    {
        delete m_layout;        //todo: sp wrap.
        m_layout = nullptr;
    }

    void setLayout(ZGraphicsLayout* pLayout) {
        m_layout = pLayout;
        m_layout->setParentItem(this);
        T::setData(GVKEY_OWNLAYOUT, QVariant::fromValue((void*)pLayout));
    }

    ZGraphicsLayout* layout() const {
        return m_layout;
    }

    void setParentLayout(ZGraphicsLayout* pLayout) {
        m_parentLayout = pLayout;
        T::setData(GVKEY_PARENT_LAYOUT, QVariant::fromValue((void*)pLayout));
    }

    ZGraphicsLayout* parentLayout() const {
        return m_parentLayout;
    }

    void setSizePolicy(const QSizePolicy& policy) {
        m_policy = policy;
        T::setData(GVKEY_SIZEPOLICY, policy);
    }

    QSizePolicy sizePolicy() const {
        return m_policy;
    }

    QGraphicsItem* graphicsItem() {
        return this;
    }

    void setMinimumSize(const QSizeF& size) {
        m_minSize = size;
        T::setData(GVKEY_SIZEHINT, size);
    }

    QSizeF minimumSize() const {
        return m_minSize;
    }

    void setFixedSize(const QSizeF& size) {
        m_minSize = size;
        T::setData(GVKEY_SIZEHINT, size);
    }

    QRectF boundingRect() const override {
        //skip QGraphicsWidget.
        if (QGraphicsProxyWidget::Type == T::type())
            return T::boundingRect();

        QSizeF sizeHint = T::data(GVKEY_SIZEHINT).toSizeF();
        QVariant varPolicy = T::data(GVKEY_SIZEPOLICY);
        QSizePolicy policy = varPolicy.value<QSizePolicy>();
        QRectF br = T::boundingRect();
        qreal w = 0, h = 0;

        if (!sizeHint.isValid())
            return br;

        if (policy.horizontalPolicy() == QSizePolicy::Expanding)
        {
            w = sizeHint.width();
        }
        else
        {
            if (sizeHint.width() > 0)
                w = sizeHint.width();
            else
                w = br.width();
        }

        if (policy.verticalPolicy() == QSizePolicy::Expanding)
        {
            h = sizeHint.height();
        }
        else
        {
            if (sizeHint.height() > 0)
                h = sizeHint.height();
            else
                h = br.height();
        }
        return QRectF(br.topLeft(), QSizeF(w, h));
    }

protected:
    QSizePolicy m_policy;
    ZGraphicsLayout* m_layout;
    ZGraphicsLayout* m_parentLayout;
    QSizeF m_minSize;
};


#endif