#ifndef __VIEW_PARAM_MODEL_H__
#define __VIEW_PARAM_MODEL_H__

#include <QtWidgets>
#include "modeldata.h"

enum VPARAM_TYPE
{
    VPARAM_ROOT,
    VPARAM_TAB,
    VPARAM_GROUP,
    VPARAM_INPUTS,
    VPARAM_PARAMETERS,
    VPARAM_OUTPUTS,
    VPARAM_PARAM,
};

enum ROLE_VPARAM
{
    ROLE_VPARAM_TYPE = Qt::UserRole + 1,
    ROLE_VPARAM_NAME,
    ROLE_VPARAM_VALUE,      //real value on idx.
    ROLE_VPARAM_IS_COREPARAM,   //is mapped from core param.
    ROLE_VAPRAM_EDITTABLE,       //edittable for name and content.
    ROLE_VPARAM_ACTIVE_TABINDEX,    //active tab index
    ROLE_VPARAM_COLLASPED,      // whether group is collasped.
};


struct VParamItem : public QStandardItem
{
    QPersistentModelIndex m_index;      //index to core param.
    PARAM_INFO m_info;

    VPARAM_TYPE vType;
    const bool m_bMappedCore;     //mapped to core param.
    bool m_bEditable;

    VParamItem(VPARAM_TYPE vType, const QString& text, bool bMapCore = false);
    VParamItem(VPARAM_TYPE vType, const QIcon& icon, const QString& text, bool bMapCore = false);

    QVariant data(int role = Qt::UserRole + 1) const override;
    void setData(const QVariant& value, int role) override;
    QStandardItem* clone() const override;
    void cloneAppend(VParamItem* pItem);
    VParamItem* getItem(const QString& uniqueName) const;
    bool operator==(VParamItem* rItem) const;
};

class ViewParamModel : public QStandardItemModel
{
    Q_OBJECT
public:
    explicit ViewParamModel(QObject* parent = nullptr);
    explicit ViewParamModel(const QString& customXml, QObject* parent = nullptr);
    QString exportUI() const;
    void clone(ViewParamModel* pModel);

public slots:
    void onParamsInserted(const QModelIndex& parent, int first, int last);
    void onParamsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    void setup(const QString& customUI);
};

#endif