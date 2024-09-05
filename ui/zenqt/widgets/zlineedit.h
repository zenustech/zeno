#ifndef __ZLINEEDIT_H__
#define __ZLINEEDIT_H__

#include <QtWidgets>
#include <zeno/core/data.h>

class ZNumSlider;
class ZTimeline;
class ZenoHintListWidget;
class ZenoFuncDescriptionLabel;

class ZLineEdit : public QLineEdit
{
    Q_OBJECT
public:
    explicit ZLineEdit(QWidget* parent = nullptr);
    explicit ZLineEdit(const QString& text, QWidget* parent = nullptr);
    void setNumSlider(const QVector<qreal>& steps);
    void setNodeIdx(const QModelIndex& index);
    void setShowingSlider(bool bShow);
    bool showingSlider();
    void setIcons(const QString& icNormal, const QString& icHover);
    void hintSelectedSetText(QString text);
    void setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabl);

protected:
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    bool eventFilter(QObject *obj, QEvent *event) override;
    void focusOutEvent(QFocusEvent* event) override;

signals:
    void btnClicked();
    void textEditFinished();

public slots:
    void sltHintSelected(QString itemSelected);
    void sltSetFocus();

private:
    void popupSlider();
    void init();

    QVector<qreal> m_steps;
    ZNumSlider* m_pSlider;
    bool m_bShowingSlider;

    bool m_bHasRightBtn;
    QPushButton *m_pButton;
    QString m_iconNormal;
    QString m_iconHover;
    bool m_bIconHover;

    bool m_bShowHintList;

    QString m_firstCandidateWord;
    QPersistentModelIndex m_nodeIdx;

    ZenoHintListWidget* m_hintlist;
    ZenoFuncDescriptionLabel* m_descLabel;
};

class ZCoreParamLineEdit : public ZLineEdit
{
    Q_OBJECT
public:
    ZCoreParamLineEdit(zeno::PrimVar var, zeno::ParamType targetType, QWidget* parent = nullptr);
    zeno::PrimVar getPrimVariant() const;

    //curve
    void setKeyFrame(const QStringList& keys);
    void delKeyFrame(const QStringList& keys);
    void editKeyFrame(const QStringList& keys);
    void clearKeyFrame(const QStringList& keys);
    void serKeyFrameStyle(QVariant qvar);

signals:
    void valueChanged(zeno::PrimVar newVal);

private:
    zeno::PrimVar m_var;
    zeno::ParamType m_targetType;
};

#endif
