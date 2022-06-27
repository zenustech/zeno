#ifndef __ZENO_PARAM_WIDGET_H__
#define __ZENO_PARAM_WIDGET_H__

#include <QtWidgets>
#include <zenoui/model/modelrole.h>
#include "../../nodesys/nodesys_common.h"
#include "zenosocketitem.h"
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/zveceditor.h>
#include <zenoui/comctrl/zcheckboxbar.h>


class ZenoParamWidget : public QGraphicsProxyWidget
{
	Q_OBJECT
public:
    ZenoParamWidget(QGraphicsItem *parent = nullptr, Qt::WindowFlags wFlags = Qt::WindowFlags());
    ~ZenoParamWidget();

    enum {
        Type = ZTYPE_PARAMWIDGET
    };
    int type() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override;

signals:
    void doubleClicked();
};

class ZenoFrame : public QFrame
{
    Q_OBJECT
public:
    ZenoFrame(QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
    ~ZenoFrame();

    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent* e) override;
};


class ZenoGvLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    ZenoGvLineEdit(QWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent *e) override;
};


class ZenoParamLineEdit : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamLineEdit(const QString &text, PARAM_CONTROL ctrl, LineEditParam param, QGraphicsItem *parent = nullptr);
    QString text() const;
    void setText(const QString& text);

signals:
    void editingFinished();

private:
    QLineEdit *m_pLineEdit;
};


class ZenoParamCheckBox : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamCheckBox(const QString &text, QGraphicsItem *parent = nullptr);
    Qt::CheckState checkState() const;
    void setCheckState(Qt::CheckState state);

Q_SIGNALS:
    void stateChanged(int);

private:
    ZCheckBoxBar* m_pCheckbox;
};


class ZenoVecEditWidget : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoVecEditWidget(const QVector<qreal>& vec, QGraphicsItem* parent = nullptr);
    QVector<qreal> vec() const;
    void setVec(const QVector<qreal>& vec);

signals:
    void editingFinished();

private:
    ZVecEditor* m_pEdit;
};

class ZenoParamLabel : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamLabel(const QString &text, const QFont& font, const QBrush& fill, QGraphicsItem *parent = nullptr);
    void setAlignment(Qt::Alignment alignment);
    void setText(const QString& text);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

private:
    QLabel* m_label;
};

class ZComboBoxItemDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    ZComboBoxItemDelegate(QObject *parent = nullptr);
    // painting
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override;

protected:
    void initStyleOption(QStyleOptionViewItem *option, const QModelIndex &index) const override;
};

class ZenoGvComboBox : public QComboBox
{
    Q_OBJECT
public:
    ZenoGvComboBox(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *e) override;
};

class ZenoParamComboBox : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamComboBox(const QStringList& items, ComboBoxParam param, QGraphicsItem *parent = nullptr);
    void setText(const QString& text);
    QString text();

signals:
    void textActivated(const QString& text);

private slots:
    void onComboItemActivated(int index);

private:
    ZComboBox* m_combobox;
};

class ZenoParamPushButton : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamPushButton(const QString& name, int width, QSizePolicy::Policy hor, QGraphicsItem* parent = nullptr);

signals:
    void clicked(bool checked = false);

private:
    int m_width;
};

class ZenoParamOpenPath : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamOpenPath(const QString& filename, QGraphicsItem* parent = nullptr);

private:
    QString m_path;
};

class ZenoParamMultilineStr : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamMultilineStr(const QString &value, LineEditParam param, QGraphicsItem *parent = nullptr);
    QString text() const;
    void setText(const QString &text);

protected:
    bool eventFilter(QObject *object, QEvent *event) override;

signals:
    void textChanged();
    void editingFinished();

private:
    void initTextFormat();

    QString m_value;
    QTextEdit* m_pTextEdit;
};

class ZenoParamBlackboard : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamBlackboard(const QString &value, LineEditParam param, QGraphicsItem *parent = nullptr);
    QString text() const;
    void setText(const QString &text);

protected:
    bool eventFilter(QObject *object, QEvent *event) override;

signals:
    void textChanged();
    void editingFinished();

private:
    QString m_value;
    QTextEdit *m_pTextEdit;
};


class ZenoSpacerItem : public QGraphicsLayoutItem, public QGraphicsItem
{
public:
    ZenoSpacerItem(bool bHorizontal, qreal size, QGraphicsItem* parent = nullptr);
    void setGeometry(const QRectF& rect) override;
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF& constraint = QSizeF()) const override;

private:
    qreal m_size;
    bool m_bHorizontal;
};

class ZenoTextLayoutItem : public QGraphicsTextItem, public QGraphicsLayoutItem
{
    Q_OBJECT
public:
    ZenoTextLayoutItem(const QString &text, const QFont &font, const QColor &color, QGraphicsItem *parent = nullptr);
    void setGeometry(const QRectF &rect) override;
    void setRight(bool right);
    void setText(const QString& text);
    void setMargins(qreal leftM, qreal topM, qreal rightM, qreal bottomM);
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    QPainterPath shape() const override;

signals:
    void editingFinished();
    void contentsChanged(QString oldText, QString newText);

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;
    void focusOutEvent(QFocusEvent *event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

private:
    void initAlignment(qreal textWidth);

    QString m_text;
    bool m_bRight;
};

class ZenoBoardTextLayoutItem : public QGraphicsLayoutItem, public QGraphicsTextItem
{
public:
    ZenoBoardTextLayoutItem(const QString& text, const QFont& font, const QColor& color, const QSizeF& sz, QGraphicsItem* parent = nullptr);
    void setGeometry(const QRectF &rect) override;
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;

private:
    QString m_text;
    QSizeF m_size;
};

class ZenoMinStatusBtnItem : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;
public:
    ZenoMinStatusBtnItem(const StatusComponent& statusComp, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
    void setChecked(STATUS_BTN btn, bool bChecked);
    void setOptions(int options);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;

signals:
    void toggleChanged(STATUS_BTN btn, bool hovered);

protected:
    ZenoImageItem* m_minMute;
    ZenoImageItem* m_minView;
    ZenoImageItem* m_minOnce;

    ZenoImageItem* m_mute;
    ZenoImageItem* m_view;
    ZenoImageItem* m_once;
};

class ZenoMinStatusBtnWidget : public QGraphicsLayoutItem, public ZenoMinStatusBtnItem
{
public:
    ZenoMinStatusBtnWidget(const StatusComponent& statusComp, QGraphicsItem* parent = nullptr);
	void updateGeometry() override;
	void setGeometry(const QRectF& rect) override;
	QRectF boundingRect() const override;

protected:
	QSizeF sizeHint(Qt::SizeHint which, const QSizeF& constraint = QSizeF()) const override;
};


class ZenoSvgLayoutItem : public QGraphicsLayoutItem, public ZenoImageItem
{
public:
    ZenoSvgLayoutItem(const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent = 0);
    void updateGeometry() override;
    void setGeometry(const QRectF &rect) override;
    QRectF boundingRect() const override;

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;
};

//Qt offical layout item demo.
class SpacerLayoutItem : public QGraphicsLayoutItem
{
public:
    SpacerLayoutItem(QSizeF sz, bool bHorizontal, QGraphicsLayoutItem *parent = nullptr, bool isLayout = false);

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;

private:
    QSizeF m_sz;
};

#endif
