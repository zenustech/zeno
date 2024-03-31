#ifndef __ZPATHEDIT_H__
#define __ZPATHEDIT_H__

#include "zlineedit.h"
#include "nodeeditor/gv/callbackdef.h"
#include <zeno/core/common.h>

class ZPathEdit : public ZLineEdit
{
    Q_OBJECT
public:
    explicit ZPathEdit(zeno::ParamControl ctrl = zeno::ReadPathEdit, QWidget *parent = nullptr);
    explicit ZPathEdit(const QString &text, zeno::ParamControl ctrl = zeno::ReadPathEdit, QWidget *parent = nullptr);
    void setPathFlag(zeno::ParamControl ctrl);
private:
  void initUI();

  zeno::ParamControl m_ctrl;
};

#endif
