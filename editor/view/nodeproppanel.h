#ifndef __NODE_PROPERTIES_PANEL_H__
#define __NODE_PROPERTIES_PANEL_H__

#include <QtWidgets>

class NodePropertyPanel : public QWidget
{
	Q_OBJECT
public:
    NodePropertyPanel(QWidget* parent = nullptr);
    ~NodePropertyPanel();

private:
    void initUI();
};

#endif