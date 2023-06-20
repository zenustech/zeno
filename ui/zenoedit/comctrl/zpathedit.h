#ifndef __ZPATHEDIT_H__
#define __ZPATHEDIT_H__

#include <zenoui/comctrl/zlineedit.h>
#include <zenoui/comctrl/gv/callbackdef.h>

class ZPathEdit : public ZLineEdit
{
    Q_OBJECT
public:
    explicit ZPathEdit(QWidget *parent = nullptr);
  explicit ZPathEdit(const QString &text, QWidget *parent = nullptr);

private:
  void initUI();
};

#endif
