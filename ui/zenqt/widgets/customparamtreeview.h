#ifndef __CUSTOM_PARAM_TREEVIEW_H__
#define __CUSTOM_PARAM_TREEVIEW_H__

#include <QTreeView>

class CustomParamTreeView : public QTreeView
{
    Q_OBJECT
    typedef QTreeView _base;
public:
    explicit CustomParamTreeView(QWidget* parent = nullptr);

protected:
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
};

#endif