#ifndef __ZENO_GRAPHS_EDITOR_H__
#define __ZENO_GRAPHS_EDITOR_H__

#include <QtWidgets>

class ZToolButton;
class ZenoSubnetListView;
class ZenoGraphsTabWidget;


class ZenoGraphsEditor : public QWidget
{
    Q_OBJECT
public:
    ZenoGraphsEditor(QWidget* parent = nullptr);
    ~ZenoGraphsEditor();

private:
    ZToolButton* m_pSubnetBtn;
    ZenoSubnetListView* m_pSubnetList;
    ZenoGraphsTabWidget* m_pTabWidget;
};

#endif