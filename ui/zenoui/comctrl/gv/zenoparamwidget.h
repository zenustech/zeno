#ifndef __ZENO_PARAM_WIDGET_H__
#define __ZENO_PARAM_WIDGET_H__

#include <QtWidgets>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/modeldata.h>
#include "../../nodesys/nodesys_common.h"
#include "zenosocketitem.h"
#include "zgraphicstextitem.h"
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/zveceditor.h>
#include <zenoui/comctrl/zcheckboxbar.h>
#include <zenoui/comctrl/zcheckbox.h>
#include <zenoui/comctrl/zlineedit.h>
#include <zenoui/comctrl/znumslider.h>
#include <zenoui/comctrl/zspinboxslider.h>


class ZenoTextLayoutItem;
class ZGraphicsNumSliderItem;

class ZenoParamWidget : public QGraphicsProxyWidget
{
	Q_OBJECT
public:
    ZenoParamWidget(QGraphicsItem *parent = nullptr, Qt::WindowFlags wFlags = Qt::WindowFlags());
    ~ZenoParamWidget();

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
    void setValidator(const QValidator* pValidator);
    void setNumSlider(QGraphicsScene* pScene, const QVector<qreal>& steps);
    void setFont(const QFont &font);

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;

signals:
    void editingFinished();

private:
    QGraphicsView* _getFocusViewByCursor();

    ZLineEdit* m_pLineEdit;
    ZGraphicsNumSliderItem* m_pSlider;
};


class ZenoSvgLayoutItem;

class ZenoParamPathEdit : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamPathEdit(const QString& path, PARAM_CONTROL ctrl, LineEditParam param, QGraphicsItem *parent = nullptr);
    QString path() const;
    void setPath(const QString& path);
    QString getOpenFileName(const QString& caption, const QString& dir, const QString& filter);
    void setValidator(QValidator*);

signals:
    void pathValueChanged(QString);
    void clicked();     //due the bug of rendering when open dialog, we have to move out this signal.

private:
    ZenoParamLineEdit* m_pLineEdit;
    ZenoSvgLayoutItem* m_openBtn;
};


class ZenoParamCheckBox : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamCheckBox(QGraphicsItem *parent = nullptr);
    Qt::CheckState checkState() const;
    void setCheckState(Qt::CheckState state);

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;

Q_SIGNALS:
    void stateChanged(int);

private:
    ZCheckBox* m_pCheckbox;
};


class ZenoVecEditItem : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoVecEditItem(const UI_VECTYPE& vec, bool bFloat, LineEditParam param, QGraphicsScene* pScene, QGraphicsItem* parent = nullptr);
    UI_VECTYPE vec() const;
    void setVec(const UI_VECTYPE& vec, bool bFloat, QGraphicsScene* pScene);
    void setVec(const UI_VECTYPE& vec);
    bool isFloatType() const;

signals:
    void editingFinished();

private:
    void initUI(const UI_VECTYPE& vec, bool bFloat, QGraphicsScene* pScene);

    QVector<ZenoParamLineEdit*> m_editors;
    LineEditParam m_param;
    bool m_bFloatVec;
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
    ZenoParamComboBox(QGraphicsItem* parent = nullptr);
    ZenoParamComboBox(const QStringList& items, ComboBoxParam param, QGraphicsItem *parent = nullptr);
    void setItems(const QStringList& items);
    void setText(const QString& text);
    QString text();

protected:
    bool eventFilter(QObject* object, QEvent* event) override;

signals:
    void textActivated(const QString& text);

private slots:
    void onComboItemActivated(int index);
    void onBeforeShowPopup();
    void onAfterHidePopup();

private:
    ZComboBox* m_combobox;
};

class ZenoParamPushButton : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamPushButton(QGraphicsItem* parent = nullptr);
    ZenoParamPushButton(const QString& name, const QString& qssName, QGraphicsItem* parent = nullptr);
    ZenoParamPushButton(const QString& name, int width, QSizePolicy::Policy hor, QGraphicsItem* parent = nullptr);
    void setText(const QString& text);

signals:
    void clicked(bool checked = false);

private:
    int m_width;
    QPushButton* m_pBtn;
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
    ZenoParamMultilineStr(QGraphicsItem* parent = nullptr);
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
    void foucusInEdit();
    void updateStyleSheet(int fontSize);

protected:
    bool eventFilter(QObject *object, QEvent *event) override;

signals:
    void textChanged();
    void editingFinished();

private:
    QString m_value;
    QTextEdit *m_pTextEdit;
};

class ZenoParamSlider : public ZenoParamWidget {
    Q_OBJECT
  public:
    ZenoParamSlider(Qt::Orientation orientation, int value, const SLIDER_INFO &info, QGraphicsItem *parent = nullptr);
    void setValue(int value);
    void setSliderInfo(const SLIDER_INFO &info);
  signals:
    void valueChanged(int);

  private:
    void updateStyleSheet();
  private:
    QSlider *m_pSlider;
};

class ZenoParamSpinBoxSlider : public ZenoParamWidget 
{
    Q_OBJECT
  public:
    ZenoParamSpinBoxSlider(Qt::Orientation orientation, int value, const SLIDER_INFO &info, QGraphicsItem *parent = nullptr);
    void setValue(int value);
    void setSliderInfo(const SLIDER_INFO &info);
  signals:
    void valueChanged(int);

  private:
    void updateStyleSheet();

  private:
    ZSpinBoxSlider *m_pSlider;
};

class ZenoParamSpinBox : public ZenoParamWidget {
    Q_OBJECT
  public:
    ZenoParamSpinBox(const SLIDER_INFO &info, QGraphicsItem *parent = nullptr);
    void setValue(int value);
    void setSliderInfo(const SLIDER_INFO &info);
  signals:
    void valueChanged(int);
  private:
    QSpinBox *m_pSpinBox;
};

class ZenoParamDoubleSpinBox : public ZenoParamWidget 
{
    Q_OBJECT
  public:
    ZenoParamDoubleSpinBox(const SLIDER_INFO &info, QGraphicsItem *parent = nullptr);
    void setValue(double value);
    void setSliderInfo(const SLIDER_INFO &info);
  signals:
    void valueChanged(double);

  private:
    QDoubleSpinBox *m_pSpinBox;
};

class ZenoParamGroupLine : public QGraphicsItem
{
public:
    ZenoParamGroupLine(const QString &text, QGraphicsItem *parent = nullptr);
   QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    void setText(const QString &text);

  private:
    QString m_text;
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
    void setBackground(const QColor& clr);
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    QPainterPath shape() const override;
    void setScalesSlider(QGraphicsScene* pScene, const QVector<qreal>& scales);

signals:
    void editingFinished();
    void contentsChanged(QString oldText, QString newText);

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;
    void focusOutEvent(QFocusEvent *event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;

private:
    void initAlignment(qreal textWidth);

    QString m_text;
    QColor m_bg;
    ZGraphicsNumSliderItem* m_pSlider;
    QVector<qreal> m_scales;
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
    void onZoomed();

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
