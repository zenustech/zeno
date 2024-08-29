#ifndef __ZVEC_EDITOR_H__
#define __ZVEC_EDITOR_H__

#include <QtWidgets>
#include "panel/ZenoHintListWidget.h"
#include "uicommon.h"
#include <zeno/core/common.h>

class ZLineEdit;


class ZVecEditor : public QWidget
{
    Q_OBJECT
public:
    ZVecEditor(const zeno::reflect::Any& vec, zeno::ParamType paramType, int deflSize, QString styleCls, QWidget* parent = nullptr);
    zeno::reflect::Any vec() const;
    bool isFloat() const;
    int getCurrentEditor();
    void updateProperties(const QVector<QString>& properties);
    void setNodeIdx(const QModelIndex& index);
    void setHintListWidget(ZenoHintListWidget* hintlist, ZenoFuncDescriptionLabel* descLabl);

signals:
    void valueChanged(zeno::reflect::Any);
    void editingFinished();

public slots:
    void setVec(const zeno::reflect::Any& vec);
    void showNoFocusLineEdits(QWidget* lineEdit);

protected:
    bool eventFilter(QObject *watched, QEvent *event);

private:
    void initUI(const zeno::reflect::Any& vec);
    void setText(const QString& text, ZLineEdit*);

    QVector<ZLineEdit*> m_editors;
    int m_deflSize;
    QString m_styleCls;
    QPersistentModelIndex m_nodeIdx;
    const zeno::ParamType m_paramType;
    bool m_bFloat;

    ZenoHintListWidget* m_hintlist;
    ZenoFuncDescriptionLabel* m_descLabel;
};


#endif