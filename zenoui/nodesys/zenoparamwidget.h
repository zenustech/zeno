#ifndef __ZENO_PARAM_WIDGET_H__
#define __ZENO_PARAM_WIDGET_H__

#include <QtWidgets>
#include "../model/modelrole.h"
#include "nodesys_common.h"
#include "zenosocketitem.h"
#include "../comctrl/zcombobox.h"


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
};

class ZenoGvLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    ZenoGvLineEdit(QWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent *e);
};

/// <summary>
/// lineEdit
/// </summary>
class ZenoParamLineEdit : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamLineEdit(const QString &text, LineEditParam param, QGraphicsItem *parent = nullptr);

private:
    ZenoGvLineEdit *m_pLineEdit;
};

class ZenoParamLabel : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamLabel(const QString &text, const QFont& font, const QBrush& fill, QGraphicsItem *parent = nullptr);
    void setAlignment(Qt::Alignment alignment);

private:
    QLabel* m_label;
};

class ZComboBoxItemDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    ZComboBoxItemDelegate(ComboBoxParam param, QObject *parent = nullptr);
    // painting
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override;

protected:
    void initStyleOption(QStyleOptionViewItem *option, const QModelIndex &index) const;

private:
    ComboBoxParam m_param;
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

private:
    ZComboBox *m_combobox;
};

class ZenoParamPushButton : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamPushButton(const QString& name, QGraphicsItem* parent = nullptr);

private:
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
    ZenoParamMultilineStr(const QString &value, QGraphicsItem *parent = nullptr);

private:
    QString m_value;
};


class ZenoTextLayoutItem : public QGraphicsLayoutItem, public QGraphicsTextItem
{
public:
    ZenoTextLayoutItem(const QString &text, const QFont &font, const QColor &color, QGraphicsItem *parent = nullptr);
    void setGeometry(const QRectF &rect) override;
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

//signals:
//    void geometrySetup(const QPointF &pos);

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;

private:
    QString m_text;
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