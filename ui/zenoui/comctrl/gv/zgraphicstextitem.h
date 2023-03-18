#ifndef __ZGVTEXTITEM_H__
#define __ZGVTEXTITEM_H__

#include <QtWidgets>
#include "zgraphicslayoutitem.h"
#include "callbackdef.h"


extern qreal editor_factor;		//temp: global editor zoom factor.

#define DEBUG_TEXTITEM 1

class ZGraphicsNumSliderItem;

class ZGraphicsTextItem : public QGraphicsTextItem
{
    Q_OBJECT
public:
    ZGraphicsTextItem(QGraphicsItem* parent = nullptr);
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
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

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
    void setText(const QString& text);
    void setRight(bool right);
    void setPadding(int left, int top, int right, int bottom);
    void setAlignment(Qt::Alignment align);
    void setFixedWidth(qreal fixedWidth);
    void setHoverCursor(Qt::CursorShape cursor);
    bool isHovered() const;
    void updateBoundingRect();
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
    static QRectF setupTextLayout(QTextLayout* layout, _padding padding, Qt::Alignment align = Qt::AlignLeft, qreal fixedWidth = -1);

    QColor m_bg;
    QRectF m_boundingRect;
    _padding m_padding;
    Qt::Alignment m_alignment;
    qreal m_fixedWidth;
    Qt::CursorShape m_hoverCursor;
    bool m_bRight;
    bool m_bHovered;
#ifdef DEBUG_TEXTITEM
    QString m_text;
#endif
};

class ZEditableTextItem : public ZGraphicsLayoutItem<ZGraphicsTextItem>
{
    Q_OBJECT
    typedef ZGraphicsLayoutItem<ZGraphicsTextItem> _base;
public:
    ZEditableTextItem(const QString& text, QGraphicsItem* parent = nullptr);
    ZEditableTextItem(QGraphicsItem* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
    void setValidator(const QValidator* pValidator);
    void setNumSlider(QGraphicsScene* pScene, const QVector<qreal>& steps);
    QString text() const;

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void focusInEvent(QFocusEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;

private slots:
    void onContentsChanged();

private:
    void initUI(const QString& text);
    QGraphicsView* _getFocusViewByCursor();

    QString m_acceptableText;

    ZGraphicsNumSliderItem* m_pSlider;
    QPointer<QValidator> m_validator;
    bool m_bFocusIn;
    bool m_bValidating;
    bool m_bShowSlider;
};

class ZenoSocketItem;

class ZSocketPlainTextItem : public ZSimpleTextItem
{
    typedef ZSimpleTextItem _base;
public:
    explicit ZSocketPlainTextItem(
        const QPersistentModelIndex& viewSockIdx,
        const QString& text,
        bool bInput,
        Callback_OnSockClicked cbSockOnClick,
        QGraphicsItem* parent = nullptr);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
    const int cSocketWidth = 16;
    const int cSocketHeight = 16;
    ZenoSocketItem* m_socket;
    const QPersistentModelIndex m_viewSockIdx;
    const bool m_bInput;
};

class ZSocketEditableItem : public ZGraphicsLayoutItem<ZGraphicsTextItem>
{
    Q_OBJECT
    typedef ZGraphicsLayoutItem<ZGraphicsTextItem> _base;
public:
    explicit ZSocketEditableItem(
        const QPersistentModelIndex& viewSockIdx,
        const QString& text,
        bool bInput,
        Callback_OnSockClicked cbSockOnClick,
        Callback_EditContentsChange cbRename,
        QGraphicsItem* parent = nullptr);

    void updateSockName(const QString& name);
    QPointF getPortPos();
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
    const int cSocketWidth = 16;
    const int cSocketHeight = 16;
    ZenoSocketItem* m_socket;
    const QPersistentModelIndex m_viewSockIdx;
    const bool m_bInput;
};

#endif
