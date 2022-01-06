#ifndef __ZCOMBOBOX_H__
#define __ZCOMBOBOX_H__

#include <QComboBox>
#include "../style/zstyleoption.h"

class ZComboBox : public QComboBox
{
	Q_OBJECT
public:
    ZComboBox(QWidget *parent = nullptr);
    ~ZComboBox();
    QSize sizeHint() const override;
    //void initUI(const ZStyleOptionComboBox& styleOption);
    void initStyleOption(ZStyleOptionComboBox* option);

protected:
    void paintEvent(QPaintEvent* event);

private:
    void init();
};

#endif