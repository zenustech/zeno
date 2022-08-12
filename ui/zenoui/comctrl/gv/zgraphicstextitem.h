#ifndef __ZGVTEXTITEM_H__
#define __ZGVTEXTITEM_H__

#include <QtWidgets>

class ZGraphicsTextItem : public QGraphicsTextItem
{
    Q_OBJECT
public:
    ZGraphicsTextItem(const QString& text, const QFont& font, const QColor& color, QGraphicsItem* parent = nullptr);
    void setText(const QString& text);
    void setMargins(qreal leftM, qreal topM, qreal rightM, qreal bottomM);
    void setBackground(const QColor& clr);
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
    QPainterPath shape() const override;

signals:
    void editingFinished();
    void contentsChanged(QString oldText, QString newText);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);

private:
    QString m_text;
    QColor m_bg;
};

class ZSimpleTextItem : public QGraphicsSimpleTextItem
{
    typedef QGraphicsSimpleTextItem base;

    struct _padding
    {
        int left;
        int top;
        int right;
        int bottom;
        _padding(int l = 0, int t = 0, int r = 0, int b = 0) : left(l), right(r), top(t), bottom(b) {}
    };

public:
    explicit ZSimpleTextItem(QGraphicsItem* parent = nullptr);
    explicit ZSimpleTextItem(const QString& text, QGraphicsItem* parent = nullptr);
    ~ZSimpleTextItem();

    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void setBackground(const QColor& clr);
    void setRight(bool right);
    void setPadding(int left, int top, int right, int bottom);
    void setAlignment(Qt::Alignment align);
    void setFixedWidth(qreal fixedWidth);
    bool isHovered() const;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget);

    static QSizeF size(const QString& text, const QFont& font, int pleft, int pTop, int pRight, int pBottom);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    static QRectF setupTextLayout(QTextLayout* layout, _padding padding, Qt::Alignment align = Qt::AlignLeft, int fixedWidth = -1);
    void updateBoundingRect();

    QColor m_bg;
    QRectF m_boundingRect;
    _padding m_padding;
    Qt::Alignment m_alignment;
    bool m_bRight;
    bool m_bHovered;
    qreal m_fixedWidth;
};

class ZSimpleTextLayoutItem : public ZSimpleTextItem, public QGraphicsLayoutItem
{
public:
    ZSimpleTextLayoutItem(const QString& text, QGraphicsItem* parent = nullptr);
    void setGeometry(const QRectF& rect) override;
    QRectF boundingRect() const override;
    QPainterPath shape() const override;

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF& constraint = QSizeF()) const override;

};



#endif