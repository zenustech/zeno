#ifndef __NODE_PROPERTIES_PANEL_H__
#define __NODE_PROPERTIES_PANEL_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>
#include <zenoui/comctrl/gv/callbackdef.h>

class IGraphsModel;

class ZExpandableSection;
class ViewParamModel;

class ZenoPropPanel : public QWidget
{
    Q_OBJECT
    struct _PANEL_CONTROL
    {
        QWidget* pControl;
        QWidget* pLabel;
        QLayout* controlLayout;
        _PANEL_CONTROL() : pControl(nullptr), pLabel(nullptr), controlLayout(nullptr) {}
    };
    typedef QMap<QString, _PANEL_CONTROL> PANEL_GROUP;

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
    bool syncAddControl(QGridLayout* pGroupLayout, QStandardItem* paramItem, int row);
    bool syncAddGroup(QVBoxLayout* pTabLayout, QStandardItem* pGroupItem, int row);
    bool syncAddTab(QTabWidget* pTabWidget, QStandardItem* pTabItem, int row);

    QPersistentModelIndex m_subgIdx;
    QPersistentModelIndex m_idx;

    ViewParamModel* m_paramsModel;
    QTabWidget* m_tabWidget;
    bool m_bReentry;
};

#endif
