#ifndef __NODE_PROPERTIES_PANEL_H__
#define __NODE_PROPERTIES_PANEL_H__

#include <QtWidgets>
#include <zenoui/model/modeldata.h>

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
        std::function<void()> fSlot;
    };

    struct _PANEL_CONTROL
    {
        QWidget* pControl;
        QWidget* pLabel;
        QLayout* controlLayout;
        _PANEL_CONTROL() : pControl(nullptr), pLabel(nullptr), controlLayout(nullptr) {}
    };

    struct PANEL_GROUP
    {
        QMap<QString, _PANEL_CONTROL> m_ctrls;
    };

public:
    ZenoPropPanel(QWidget* parent = nullptr);
    ~ZenoPropPanel();
    void reset(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select);
    virtual QSize sizeHint() const override;
    virtual QSize minimumSizeHint() const override;

protected:
    void mousePressEvent(QMouseEvent* event) override;

public slots:
    void onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role);
    void onParamEditFinish();
    void onInputEditFinish();

private:
    ZExpandableSection* paramsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes);
    ZExpandableSection* inputsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes);
    QWidget* initControl(CONTROL_DATA ctrlData);
    void clearLayout();
    bool isMatchControl(PARAM_CONTROL ctrl, QWidget* pControl);
    void updateControlValue(QWidget* pControl, PARAM_CONTROL ctrl, const QVariant& value);
    void onInputsCheckUpdate();
    void onParamsCheckUpdate();
    void onGroupCheckUpdated(const QString& groupName, const QMap<QString, CONTROL_DATA>& ctrls);

    QPersistentModelIndex m_subgIdx;
    QPersistentModelIndex m_idx;

    QMap<QString, PANEL_GROUP> m_groups;

    QMap<QString, _PANEL_CONTROL> m_inputsCtrl;
    QMap<QString, _PANEL_CONTROL> m_paramsCtrl;

    bool m_bReentry;
};

#endif
