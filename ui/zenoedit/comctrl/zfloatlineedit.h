#ifndef __ZFLOATLINEEDIT_H__
#define __ZFLOATLINEEDIT_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>
#include <zenoui/comctrl/zlineedit.h>

class ZTimeline;
struct CURVE_DATA;
class ZFloatLineEdit : public ZLineEdit {
    Q_OBJECT
  public:
    explicit ZFloatLineEdit(QWidget *parent = nullptr);
    explicit ZFloatLineEdit(const QString &text, QWidget *parent = nullptr);
    void updateCurveData();

  protected:
    bool event(QEvent *event) override;

  private slots:
    void updateBackgroundProp(int frame);
    void onUpdate(bool gl, int frame);

  private:
    ZTimeline *getTimeline();
};

#endif
