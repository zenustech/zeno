#ifndef __ZENO_GRAPHS_EDITOR_H__
#define __ZENO_GRAPHS_EDITOR_H__

#include <QtWidgets>

class ZToolButton;
class ZenoSubnetListView;
class ZenoGraphsTabWidget;
class ZenoSubnetListPanel;

class ZenoGraphsEditor : public QWidget
{
    Q_OBJECT
public:
    ZenoGraphsEditor(QWidget* parent = nullptr);
    ~ZenoGraphsEditor();

public slots:
    void onModelInited();

private slots:
    void onSubnetBtnClicked();
    void onListItemActivated(const QModelIndex& index);

private:
    ZToolButton* m_pSubnetBtn;
    ZenoSubnetListPanel* m_pSubnetList;
    ZenoGraphsTabWidget* m_pTabWidget;
};

#endif