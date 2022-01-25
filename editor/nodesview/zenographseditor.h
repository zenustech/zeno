#ifndef __ZENO_GRAPHS_EDITOR_H__
#define __ZENO_GRAPHS_EDITOR_H__

#include <QtWidgets>

class ZToolButton;
class ZenoSubnetListView;
class ZenoGraphsTabWidget;
class ZenoSubnetListPanel;
class GraphsModel;

class ZenoGraphsEditor : public QWidget
{
    Q_OBJECT
public:
    ZenoGraphsEditor(QWidget* parent = nullptr);
    ~ZenoGraphsEditor();
    void resetModel(GraphsModel* pModel);

public slots:
    void onCurrentModelClear();

private slots:
    void onSubnetBtnClicked();
    void onListItemActivated(const QModelIndex& index);

private:
    QWidget* m_pSideBar;
    ZToolButton* m_pSubnetBtn;
    ZenoSubnetListPanel* m_pSubnetList;
    ZenoGraphsTabWidget* m_pTabWidget;
};

#endif