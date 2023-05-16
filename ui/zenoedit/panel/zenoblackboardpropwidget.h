#ifndef __ZENO_BLACKBOARD_PROP_WIDGET_H__
#define __ZENO_BLACKBOARD_PROP_WIDGET_H__

#include <QWidget>
#include <zenomodel/include/modeldata.h>
#include "comctrl/ztextedit.h"
#include "comctrl/zlineedit.h"

class ZenoBlackboardPropWidget : public QWidget 
{
    Q_OBJECT
  public:
    ZenoBlackboardPropWidget(const QPersistentModelIndex &index, const QPersistentModelIndex &subIndex, QWidget *parent = nullptr);
    ~ZenoBlackboardPropWidget();

  private:
    void insertRow(const QString &desc, const PARAM_CONTROL &type, const QVariant &value, int row, QGridLayout *pGroupLayout);
  private slots:
    void onDataChanged(const QModelIndex &, const QModelIndex &, int role);
  private:
    QPersistentModelIndex m_subgIdx;
    QPersistentModelIndex m_idx;
    ZTextEdit *m_pTitle;
    //ZTextEdit *m_pTextEdit;
    QPushButton* m_pColor;
};
#endif
