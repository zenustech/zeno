#ifndef __ZPROPERTIES_PANEL_H__
#define __ZPROPERTIES_PANEL_H__

class ValueInputWidget : public QWidget
{
    Q_OBJECT
public:
    ValueInputWidget(const QString& name, QWidget* parent = nullptr);
    void setValue(qreal value);
    qreal value(bool& bOK);

signals:
    void valueChanged();

private:
    QSpinBox* m_pSpinBox;
    QLineEdit* m_pLineEdit;
};

class ImageGroupBox : public QGroupBox
{
    Q_OBJECT
public:
    ImageGroupBox(QWidget *parent = nullptr);

signals:
    void normalImported(QString);
    void hoverImported(QString);
    void selectedImported(QString);

private:
    QLabel *m_pNormal;
    QLabel *m_pHovered;
    QLabel *m_pSelected;

    QString m_normal;
    QString m_hovered;
    QString m_selected;
};

class ColorWidget : public QWidget
{
    Q_OBJECT
public:
    ColorWidget(QWidget* parent = nullptr);
    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event);
    void mouseReleaseEvent(QMouseEvent *event);

signals:
    void colorChanged(QColor color);

private:
    QColor m_color;
};

class TextGroupBox : public QGroupBox
{
    Q_OBJECT
public:
    TextGroupBox(QWidget *parent = nullptr);

signals:
    void fontChanged(QFont font, QColor color);
    void textChanged(QString text);

private slots:
    void onValueChanged(int);

private:
    int m_fontsize;
    QFont m_font;
    QColor m_color;
    ColorWidget* m_colorWidget;
};

class TransformGroupBox : public QGroupBox
{
    Q_OBJECT
public:
    TransformGroupBox(QWidget *parent = nullptr);
    void setValue(const qreal& x, const qreal& y, const qreal& w, const qreal& h);
    bool getValue(qreal& x, qreal& y, qreal& w, qreal& h);

signals:
    void valueChanged();

private:
    ValueInputWidget *m_pWidth;
    ValueInputWidget *m_pHeight;
    ValueInputWidget *m_pX;
    ValueInputWidget *m_pY;
};

class ZPagePropPanel : public QWidget
{
    Q_OBJECT
public:
    ZPagePropPanel(QWidget* parent = nullptr);

private:
    ValueInputWidget* m_pWidth;
    ValueInputWidget* m_pHeight;
};

class ZComponentPropPanel : public QWidget
{
    Q_OBJECT
public:
    ZComponentPropPanel(QWidget* parent = nullptr);
    void initModel();

public slots:
    void onModelDataChanged(QStandardItem* pItem);
    void onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected);

private:
    void onUpdateModel(QStandardItemModel* model, QItemSelectionModel* selection);

    TransformGroupBox *m_pGbTransform;
    ImageGroupBox *m_pGbImage;
    TextGroupBox* m_pGbText;
};

class ZElementPropPanel : public QWidget
{
    Q_OBJECT
public:
    ZElementPropPanel(QWidget* parent = nullptr);

private:
    QLabel* m_pAsset;
    ValueInputWidget* m_pWidth;
    ValueInputWidget* m_pHeight;
    ValueInputWidget* m_pX;
    ValueInputWidget* m_pY;
};


#endif