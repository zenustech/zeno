#ifndef __ZENO_GRAPHS_EDITOR_H__
#define __ZENO_GRAPHS_EDITOR_H__

#include <QtWidgets>

class ZToolButton;
class ZenoSubnetListView;
class ZenoGraphsTabWidget;
class ZenoSubnetPanel;
class ZenoGraphsLayerWidget;
class ZenoSubnetTreeView;
class ZenoWelcomePage;
class ZenoMainWindow;
class IGraphsModel;

class ZenoGraphsEditor : public QWidget
{
    Q_OBJECT
public:
    ZenoGraphsEditor(ZenoMainWindow* pMainWin);
    ~ZenoGraphsEditor();
    QString getCurrentFileName();

signals:
    void modelLoaded(const QString& fn);

public slots:
    void resetModel(IGraphsModel* pModel);
    void onCurrentModelClear();
    void onItemActivated(const QModelIndex& index);
    void onPageActivated(const QPersistentModelIndex& subgIdx, const QPersistentModelIndex& nodeIdx);
    void onGraphsItemInserted(const QModelIndex& parent, int first, int last);
    void onGraphsItemAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private slots:
    void onSubnetBtnClicked();
    void onViewBtnClicked();

private:
    QWidget* m_pSideBar;
    ZToolButton* m_pSubnetBtn;
    ZToolButton* m_pViewBtn;
    ZenoSubnetPanel* m_pSubnetPanel;
    ZenoGraphsTabWidget* m_pTabWidget;
    ZenoGraphsLayerWidget* m_pLayerWidget;
    ZenoWelcomePage* m_welcomePage;
    ZenoMainWindow* m_mainWin;
    bool m_bListView;
};

#endif
