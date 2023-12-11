#ifndef __ZVEC_EDITOR_ITEM_H__
#define __ZVEC_EDITOR_ITEM_H__

#include "zgraphicslayoutitem.h"
#include "zenoparamwidget.h"

class ZVecEditorItem : public ZGraphicsLayoutItem<ZenoParamWidget>
{
    typedef ZGraphicsLayoutItem<ZenoParamWidget> _base;
    Q_OBJECT
public:
    ZVecEditorItem(const QVariant& vec, bool bFloat, LineEditParam param, QGraphicsScene* pScene, QGraphicsItem* parent = nullptr, Qt::WindowFlags wFlags = Qt::WindowFlags());
    QVariant vec() const;
    UI_VECSTRING vecStr() const;
    void setVec(const QVariant &vec, bool bFloat, QGraphicsScene *pScene);
    void setVec(const QVariant &vec);
    bool isFloatType() const;
    void updateProperties(const QVector<QString>& properties);
    QString findElemByControl(ZEditableTextItem* pElem) const;
    bool hasSliderShow();

signals:
    void editingFinished();

private:
    void initUI(const QVariant &vec, bool bFloat, QGraphicsScene *pScene);

    QVector<ZEditableTextItem*> m_editors;
    LineEditParam m_param;
    bool m_bFloatVec;
};

#endif