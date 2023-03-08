#ifndef __NODE_PROPERTIES_PANEL_H__
#define __NODE_PROPERTIES_PANEL_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>
#include <zenoui/comctrl/gv/callbackdef.h>

class IGraphsModel;

class ZExpandableSection;

class ZenoPropPanel : public QWidget
{
    Q_OBJECT
    struct _PANEL_CONTROL
    {
        QWidget* pControl;
        QLabel* pLabel;
        QLabel *pIcon;
        QLayout* controlLayout;
        QPersistentModelIndex m_viewIdx;    //compare when rename.
        _PANEL_CONTROL() : pControl(nullptr), pLabel(nullptr), pIcon(nullptr), controlLayout(nullptr) {}
    };
    typedef FuckQMap<QString, _PANEL_CONTROL> PANEL_GROUP;
    typedef FuckQMap<QString, PANEL_GROUP> PANEL_TAB;
    typedef FuckQMap<QString, PANEL_TAB> PANEL_TABS;

public:
    ZenoPropPanel(QWidget* parent = nullptr);
    ~ZenoPropPanel();
    void reset(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    virtual QSize sizeHint() const override;
    virtual QSize minimumSizeHint() const override;

public slots:
    void onViewParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void onViewParamInserted(const QModelIndex& parent, int first, int last);
    void onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void onSettings();

private:
    void clearLayout();
    bool syncAddControl(ZExpandableSection* pGroupWidget, QGridLayout* pGroupLayout, QStandardItem* paramItem, int row);
    bool syncAddGroup(QVBoxLayout* pTabLayout, QStandardItem* pGroupItem, int row);
    bool syncAddTab(QTabWidget* pTabWidget, QStandardItem* pTabItem, int row);
    ZExpandableSection* findGroup(const QString& tabName, const QString& groupName);

    QPersistentModelIndex m_subgIdx;
    QPersistentModelIndex m_idx;

    QTabWidget* m_tabWidget;
    bool m_bReentry;

    PANEL_TABS m_controls;
};

#endif
