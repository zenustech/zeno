#ifndef __ZGVFLOATTEXTITEM_H__
#define __ZGVFLOATTEXTITEM_H__

#include <QtWidgets>
#include <zenoui/comctrl/gv/zgraphicstextitem.h>

struct CURVE_DATA;
class ZFloatEditableTextItem : public ZEditableTextItem {
    Q_OBJECT
    typedef ZEditableTextItem _base;

  public:
    ZFloatEditableTextItem(const QString &text, QGraphicsItem *parent = nullptr);
    ZFloatEditableTextItem(QGraphicsItem *parent = nullptr);

  protected:
    bool event(QEvent *event) override;

  private:
    void updateCurveData();
  private slots:
    void updateText(int frame);
    void onUpdate(bool gl, int frame);
};
#endif
