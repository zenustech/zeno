#ifndef __ZVEC_EDITOR_H__
#define __ZVEC_EDITOR_H__

#include <QtWidgets>
#include "panel/ZenoHintListWidget.h"
#include "uicommon.h"
#include <zeno/core/common.h>

class ZLineEdit;
class ZTextEdit;

class ZVecEditor : public QWidget
{
    Q_OBJECT
public:
    ZVecEditor(const zeno::vecvar& vec, bool bFloat, QString styleCls, QWidget* parent = nullptr);
    zeno::vecvar vec() const;
    bool isFloat() const;
    int getCurrentEditor();
    void updateProperties(const QVector<QString>& properties);
    void setNodeIdx(const QModelIndex& index);
    void setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabl);

    //keyframe
    void setKeyFrame(const QStringList& keys);
    void delKeyFrame(const QStringList& keys);
    void editKeyFrame(const QStringList& keys);
    void clearKeyFrame(const QStringList& keys);
    bool serKeyFrameStyle(QVariant qvar);

signals:
    void valueChanged(zeno::vecvar);
    void editingFinished();

public slots:
    void setVec(const zeno::vecvar& vec, bool bFloat);
    void showNoFocusLineEdits(QWidget* lineEdit);

protected:
    bool eventFilter(QObject *watched, QEvent *event);
    void resizeEvent(QResizeEvent* event) override;

private:
    void initUI(const zeno::vecvar& vec);
    void setText(const QString& text, ZLineEdit*);
    void showMultiLineEdit(int i);

    zeno::vecvar m_vec;
    QVector<ZLineEdit*> m_editors;
    int m_deflSize;
    QString m_styleCls;
    QPersistentModelIndex m_nodeIdx;
    bool m_bFloat;

    ZenoHintListWidget* m_hintlist;
    ZenoFuncDescriptionLabel* m_descLabel;
    ZTextEdit* m_textEdit;
};


#endif