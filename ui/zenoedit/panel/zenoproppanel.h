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

    struct CONTROL_DATA
    {
        QString name;
        PARAM_CONTROL ctrl;
        QVariant value;
        QString typeDesc;
        bool bkFrame;
        Callback_EditFinished cbFunc;
    };

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
    void onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);

private:
    ZExpandableSection* paramsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes);
    ZExpandableSection* inputsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes);
    void clearLayout();
    void onInputsCheckUpdate();
    void onParamsCheckUpdate();
    void onGroupCheckUpdated(const QString& groupName, const QMap<QString, CONTROL_DATA>& ctrls);

    QPersistentModelIndex m_subgIdx;
    QPersistentModelIndex m_idx;

    QMap<QString, PANEL_GROUP> m_groups;

    bool m_bReentry;
};

#endif
