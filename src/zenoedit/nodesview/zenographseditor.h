#ifndef __ZENO_GRAPHS_EDITOR_H__
#define __ZENO_GRAPHS_EDITOR_H__

#include <QtWidgets>

class ZToolButton;
class ZenoSubnetListView;
class ZenoGraphsTabWidget;
class ZenoSubnetListPanel;
class ZenoGraphsLayerWidget;
class ZenoSubnetTreeView;
class IGraphsModel;

class ZenoGraphsEditor : public QWidget
{
    Q_OBJECT
public:
    ZenoGraphsEditor(QWidget* parent = nullptr);
    ~ZenoGraphsEditor();
    void resetModel(IGraphsModel* pModel);

public slots:
    void onCurrentModelClear();
    void onItemActivated(const QModelIndex& index);
    void onItemActivated(const QString& subGraphName);

private slots:
    void onSubnetBtnClicked();
    void onViewBtnClicked();

private:
    QWidget* m_pSideBar;
    ZToolButton* m_pSubnetBtn;
    ZToolButton* m_pViewBtn;
    ZenoSubnetListPanel* m_pSubnetList;
    ZenoGraphsTabWidget* m_pTabWidget;
    ZenoGraphsLayerWidget* m_pLayerWidget;
    bool m_bListView;
};

#endif
