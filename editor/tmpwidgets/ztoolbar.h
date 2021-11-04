#ifndef __ZTOOLBAR_H__
#define __ZTOOLBAR_H__

#include <QtWidgets>

class ZShapeBar : public QWidget
{
    Q_OBJECT
public:
    ZShapeBar(QWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent* e);

private:

};

class ZTextureBar : public QWidget
{
    Q_OBJECT
public:
    ZTextureBar(QWidget* parent = nullptr);

protected:
	void paintEvent(QPaintEvent* e);
};

class ZToolbar : public QWidget
{
    Q_OBJECT
public:
    ZToolbar(QWidget* parent = nullptr);
};

#endif