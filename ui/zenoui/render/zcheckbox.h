#ifndef __ZCHECKBOX_H__
#define __ZCHECKBOX_H__

#include <QtWidgets>

class ZCheckBox : public QCheckBox
{
    Q_OBJECT
public:
    explicit ZCheckBox(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;

private:

};

#endif