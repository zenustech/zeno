#ifndef __ZTOOLTIP_H__
#define __ZTOOLTIP_H__

#include <QtWidgets>

class ZToolTip : public QLabel
{
    Q_OBJECT
public:
    static ZToolTip* getInstance();
    static void showText(QPoint pos, const QString& text);
    static void showIconText(QString icon, QPoint pos, const QString& text);
    static void hideText();
protected:
    void paintEvent(QPaintEvent* evt) override;
private:
    ZToolTip(QWidget* parent = nullptr);
};

#endif