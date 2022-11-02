#ifndef __VIEW_PARAM_MODEL_H__
#define __VIEW_PARAM_MODEL_H__

#include <QtWidgets>
#include "modeldata.h"

enum VPARAM_TYPE
{
    VPARAM_ROOT,
    VPARAM_DEFAULT_TAB,
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
};


struct VParamItem : public QStandardItem
{
    QPersistentModelIndex m_index;      //index to real param.
    QString name;       //display name.
    PARAM_CONTROL ctrl;
    VPARAM_TYPE vType;

    VParamItem(VPARAM_TYPE vType, const QString& text);
    VParamItem(VPARAM_TYPE vType, const QIcon& icon, const QString& text);
    VParamItem(VPARAM_TYPE vType);

    QVariant data(int role = Qt::UserRole + 1) const override;
};

class ViewParamModel : public QStandardItemModel
{
    Q_OBJECT
public:
    explicit ViewParamModel(QObject* parent = nullptr);
    explicit ViewParamModel(const QString& customXml, QObject* parent = nullptr);
    QString exportUI() const;

public slots:
    void onParamsInserted(const QModelIndex& parent, int first, int last);
    void onParamsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    void setup(const QString& customUI);
};

#endif