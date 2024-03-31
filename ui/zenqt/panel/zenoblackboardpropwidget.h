#ifndef __ZENO_BLACKBOARD_PROP_WIDGET_H__
#define __ZENO_BLACKBOARD_PROP_WIDGET_H__

#include <QWidget>
#include "widgets/ztextedit.h"
#include "widgets/zlineedit.h"
#include "uicommon.h"

class ZenoBlackboardPropWidget : public QWidget 
{
    Q_OBJECT
  public:
    ZenoBlackboardPropWidget(const QPersistentModelIndex &index, QWidget *parent = nullptr);
    ~ZenoBlackboardPropWidget();

  private:
    void insertRow(const QString &desc, const zeno::ParamControl&type, const QVariant &value, int row, QGridLayout *pGroupLayout);
  private slots:
    void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
  private:
    QPersistentModelIndex m_idx;
    ZTextEdit *m_pTitle;
    QPushButton* m_pColor;
};
#endif
