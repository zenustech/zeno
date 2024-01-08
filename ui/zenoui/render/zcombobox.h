#ifndef __ZCOMBOBOX_H__
#define __ZCOMBOBOX_H__

#include <QComboBox>
#include "../style/zstyleoption.h"

class ZComboBox : public QComboBox
{
	Q_OBJECT
public:
    ZComboBox(bool bSysStyle = true, QWidget *parent = nullptr);
    ~ZComboBox();
    QSize sizeHint() const override;
    void initStyleOption(ZStyleOptionComboBox* option);
    void showPopup() override;
    void hidePopup() override;

signals:
    void beforeShowPopup();
    void afterHidePopup();
    void _textActivated(const QString&);

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private slots:
    void onComboItemActivated(int index);

private:
    bool m_bSysStyle;
};

#endif
