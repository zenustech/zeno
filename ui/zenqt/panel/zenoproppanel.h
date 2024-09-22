#ifndef __NODE_PROPERTIES_PANEL_H__
#define __NODE_PROPERTIES_PANEL_H__

#include <QtWidgets>
#include "uicommon.h"
#include "nodeeditor/gv/callbackdef.h"
#include "ZenoHintListWidget.h"
#include "widgets/zlineedit.h"
#include "widgets/zlabel.h"

class IGraphsModel;

class ZExpandableSection;
class ZScrollArea;
class ZenoDictListLinksTable;

class ZenoPropPanel : public QWidget
{
    Q_OBJECT
    struct _PANEL_CONTROL
    {
        QWidget* pControl;
        QLabel* pLabel;
        QLabel* pIconLabel;
        QLayout* controlLayout;
        QPersistentModelIndex m_viewIdx;    //compare when rename.
        _PANEL_CONTROL() : pControl(nullptr), pLabel(nullptr), pIconLabel(nullptr), controlLayout(nullptr) {}
    };
    typedef QKeyList<QString, _PANEL_CONTROL> PANEL_GROUP;
    typedef QKeyList<QString, PANEL_GROUP> PANEL_TAB;
    typedef QKeyList<QString, PANEL_TAB> PANEL_TABS;

public:
    ZenoPropPanel(QWidget* parent = nullptr);
    ~ZenoPropPanel();
    void reset(GraphModel* subgraph, const QModelIndexList& nodes, bool select);
    virtual QSize sizeHint() const override;
    virtual QSize minimumSizeHint() const override;
    bool updateCustomName(const QString& value, QString& oldValue);

    ZenoHintListWidget* getHintListInstance();
    ZenoFuncDescriptionLabel* getFuncDescriptionInstance();

public slots:
    void onCustomParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
    void onViewParamInserted(const QModelIndex& parent, int first, int last);
    void onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last);
    void onViewParamsMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int destRow);
    //dict/list links table
    void onLinkAdded(const zeno::EdgeInfo& link);
    void onLinkRemoved(const zeno::EdgeInfo& link);
    void onDictListTableUpdateLink(QList<QPair<QString, QModelIndex>> links);
    void onDictListTableRemoveLink(QList<QModelIndex> links);

    void onSettings();

protected:
    bool eventFilter(QObject* obj, QEvent* event);
    void paintEvent(QPaintEvent* event) override;

private slots:
    void onNodeRemoved(QString nodeName);
    void onUpdateParamsVisbleEnabled();

private:
    void clearLayout();
    bool syncAddControl(ZExpandableSection* pGroupWidget, QGridLayout* pGroupLayout, QStandardItem* paramItem, int row);
    bool syncAddGroup(QVBoxLayout* pTabLayout, QStandardItem* pGroupItem, int row);
    bool syncAddTab(QTabWidget* pTabWidget, QStandardItem* pTabItem, int row);
    ZExpandableSection* findGroup(const QString& tabName, const QString& groupName);
    void setKeyFrame(const _PANEL_CONTROL& ctrl, const QStringList& keys);
    void delKeyFrame(const _PANEL_CONTROL& ctrl, const QStringList& keys);
    void editKeyFrame(const _PANEL_CONTROL& ctrl, const QStringList& keys);
    void clearKeyFrame(const _PANEL_CONTROL& ctrl, const QStringList& keys);
    void onUpdateFrame(QWidget* pContrl, int nFrame, QVariant val);
    void normalNodeAddInputWidget(ZScrollArea* scrollArea, QGridLayout* pLayout, QStandardItem* pItem, int row);
    void addOutputWidget(ZScrollArea* scrollArea, QGridLayout* pLayout, QStandardItem* pOutputItem, int row);


    GraphModel* m_model;
    QPersistentModelIndex m_idx;

    QWidget* m_normalNodeInputWidget;       //普通节点input使用
    QTabWidget* m_tabWidget;                //子图节点input使用
    QWidget* m_outputWidget;                //output

    ZenoDictListLinksTable* m_dictListLinksTable;   //显示makeDict/makeList输入边

    bool m_bReentry;

    PANEL_TABS m_inputControls;
    PANEL_GROUP m_outputControls;
    QList<_PANEL_CONTROL> m_floatControls;

    QScopedPointer<ZenoHintListWidget> m_hintlist;
    QScopedPointer<ZenoFuncDescriptionLabel> m_descLabel;
};

#endif
