#ifndef __NODE_PROPERTIES_PANEL_H__
#define __NODE_PROPERTIES_PANEL_H__

#include <QtWidgets>

class ZenoPropPanel : public QWidget
{
	Q_OBJECT
public:
    ZenoPropPanel(QWidget* parent = nullptr);
    ~ZenoPropPanel();

private:
    void init();
};

#endif