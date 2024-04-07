#ifndef __ZENO_GRAPHS_EDITOR_H__
#define __ZENO_GRAPHS_EDITOR_H__

#include <QtWidgets>
#include "nodeeditor/gv/zenosubgraphview.h"

class ZToolButton;
class ZenoWelcomePage;
class ZenoMainWindow;
class ZenoSubGraphView;
class GraphModel;

namespace Ui
{
    class GraphsEditor;
}

class ZenoGraphsEditor : public QWidget
{
    Q_OBJECT
  public:
    enum SideBarItem
    {
        Side_Subnet,
        Side_Tree,
        Side_Search
    };

public:
    ZenoGraphsEditor(ZenoMainWindow* pMainWin);
    ~ZenoGraphsEditor();
    //void activateTab(const QString& subGraphName, const QString& path = "", const QString& objId = "", bool isError = false);
    void activateTab(const QStringList& objpath, const QString& focusNode = "", bool isError = false);
    void showFloatPanel(GraphModel* subgraph, const QModelIndexList &nodes);
    void selectTab(const QString& subGraphName, const QString& path, std::vector<QString>& objId);
    ZenoSubGraphView* getCurrentSubGraphView();

    void showWelcomPage();
    bool welComPageShowed();

public slots:
    void resetMainModel();
    void resetAssetsModel();
    void sideButtonToggled(bool bToggled);
    void onSideBtnToggleChanged(const QItemSelection& selected, const QItemSelection& deselected);
    void onCurrentChanged(const QModelIndex& current, const QModelIndex& previous);
    void onAssetItemActivated(const QModelIndex& index);
    void onTreeItemActivated(const QModelIndex& index);
    void onSearchItemClicked(const QModelIndex& index);
    void onAssetOptionClicked();
    void onSearchOptionClicked();
    void onPageActivated(const QPersistentModelIndex& subgIdx, const QPersistentModelIndex& nodeIdx);
    void onPageActivated(const QModelIndex& subgNodeIdx);
    void onLogInserted(const QModelIndex& parent, int first, int last);
    void onSubnetListPanel(bool bShow, SideBarItem item);
    void onAction(QAction* pAction, const QVariantList& args = QVariantList(), bool bChecked = false);
    void onCommandDispatched(QAction* pAction, bool bTriggered);
    void onTreeItemSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
    void onAssetsCustomParamsClicked(const QString& assetsName);

signals:
    void zoomed(qreal);

private slots:
	void onSubGraphsToRemove(const QModelIndex&, int, int);
    void onAssetsToRemove(const QModelIndex& parent, int first, int last);
	void onModelReset();
    void onModelCleared();
	void onSubGraphRename(const QString& oldName, const QString& newName);
    void onSearchEdited(const QString& content);
    void onMenuActionTriggered(QAction* pAction);
    void onNewAsset();
    void onPageListClicked();

private:
    void initUI();
    void initSignals();
    void initRecentFiles();
    void initModel();
    void toggleViewForSelected(bool bOn);
    int tabIndexOfName(const QString& subGraphName);
    void markSubgError(const QStringList &lst);
    void closeMaterialTab();

    ZenoMainWindow* m_mainWin;
    Ui::GraphsEditor* m_ui;
    //IGraphsModel* m_model;
    QItemSelectionModel* m_selection;
    QStandardItemModel* m_sideBarModel;
    int m_searchOpts;
};


#endif
