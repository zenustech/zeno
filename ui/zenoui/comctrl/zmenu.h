#ifndef __ZMENU_H__
#define __ZMENU_H__

#include <QtWidgets>

class ZMenu : public QMenu
{
    Q_OBJECT
public:
    ZMenu(QWidget* parent = nullptr);
    ZMenu(const QString& title, QWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    
};


#endif
