#ifndef __ZENO_GRAPHS_EDITOR_H__
#define __ZENO_GRAPHS_EDITOR_H__

#include <QtWidgets>
#include <zenoui/include/common.h>

class ZToolButton;
class ZenoWelcomePage;
class ZenoMainWindow;
class IGraphsModel;

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
    void activateTab(const QString& subGraphName, const QString& path = "", const QString& objId = "", bool isError = false);
    void showFloatPanel(const QModelIndex &subgIdx, const QModelIndexList &nodes);

public slots:
	void resetModel(IGraphsModel* pModel);
    void sideButtonToggled(bool bToggled);
    void onSideBtnToggleChanged(const QItemSelection& selected, const QItemSelection& deselected);
    void onCurrentChanged(const QModelIndex& current, const QModelIndex& previous);
    void onListItemActivated(const QModelIndex& index);
    void onTreeItemActivated(const QModelIndex& index);
    void onSearchItemClicked(const QModelIndex& index);
    void onSubnetOptionClicked();
    void onSearchOptionClicked();
    void onPageActivated(const QPersistentModelIndex& subgIdx, const QPersistentModelIndex& nodeIdx);
    void onLogInserted(const QModelIndex& parent, int first, int last);
    void onSubnetListPanel(bool bShow, SideBarItem item);
    void onAction(QAction* pAction, const QVariantList& args = QVariantList(), bool bChecked = false);
    void onCommandDispatched(QAction* pAction, bool bTriggered);

signals:
    void zoomed(qreal);

private slots:
	void onSubGraphsToRemove(const QModelIndex&, int, int);
	void onModelReset();
    void onModelCleared();
	void onSubGraphRename(const QString& oldName, const QString& newName);
    void onSearchEdited(const QString& content);
    void onMenuActionTriggered(QAction* pAction);

private:
    void initUI();
    void initSignals();
    void initRecentFiles();
    void initModel();
    void toggleViewForSelected(bool bOn);
    int tabIndexOfName(const QString& subGraphName);

    ZenoMainWindow* m_mainWin;
    Ui::GraphsEditor* m_ui;
    IGraphsModel* m_model;
    QItemSelectionModel* m_selection;
    QStandardItemModel* m_sideBarModel;
    int m_searchOpts;
};


#endif
