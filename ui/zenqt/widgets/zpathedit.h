#ifndef __ZPATHEDIT_H__
#define __ZPATHEDIT_H__

#include "zlineedit.h"
#include "nodeeditor/gv/callbackdef.h"

class ZPathEdit : public ZLineEdit
{
    Q_OBJECT
public:
    explicit ZPathEdit(const CALLBACK_SWITCH& cbSwitch, QWidget *parent = nullptr);
  explicit ZPathEdit(const CALLBACK_SWITCH& cbSwitch, const QString &text, QWidget *parent = nullptr);

private:
  void initUI(const CALLBACK_SWITCH& cbSwitch);
};

#endif
