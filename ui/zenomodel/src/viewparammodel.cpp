#include "viewparammodel.h"
#include "parammodel.h"
#include "zassert.h"


VParamItem::VParamItem(VPARAM_TYPE type, const QString& text)
    : QStandardItem(text)
    , ctrl(CONTROL_NONE)
    , vType(type)
    , name(text)
{
}

VParamItem::VParamItem(VPARAM_TYPE type, const QIcon& icon, const QString& text)
    : QStandardItem(icon, text)
    , ctrl(CONTROL_NONE)
    , vType(type)
    , name(text)
{
}

VParamItem::VParamItem(VPARAM_TYPE type)
    : ctrl(CONTROL_NONE)
    , vType(type)
{
}

QVariant VParamItem::data(int role) const
{
    switch (role)
    {
    case Qt::DisplayRole:
    case ROLE_VPARAM_NAME:  return name;
    case ROLE_VPARAM_TYPE:  return vType;
    case ROLE_PARAM_CTRL:   return ctrl;
    default:
        return QVariant();
    }
}



ViewParamModel::ViewParamModel(QObject* parent)
    : QStandardItemModel(parent)
{
    setup("");
}

ViewParamModel::ViewParamModel(const QString& customXml, QObject* parent)
    : QStandardItemModel(parent)
{
    setup(customXml);
}

void ViewParamModel::setup(const QString& customUI)
{
    if (customUI.isEmpty())
    {
        /*default structure:
            root
                |-- Tab (Default)
                    |-- Inputs (Group)
                        -- input param1 (Item)
                        -- input param2
                        ...

                    |-- Params (Group)
                        -- param1 (Item)
                        -- param2 (Item)
                        ...

                    |- Outputs (Group)
                        - output param1 (Item)
                        - output param2 (Item)
            ...
        */
        VParamItem* pRoot = new VParamItem(VPARAM_ROOT, "root");

        VParamItem* pTab = new VParamItem(VPARAM_DEFAULT_TAB, "Default");
        {
            VParamItem* pInputsGroup = new VParamItem(VPARAM_GROUP, "In Sockets");
            VParamItem* paramsGroup = new VParamItem(VPARAM_GROUP, "Parameters");
            VParamItem* pOutputsGroup = new VParamItem(VPARAM_GROUP, "Out Sockets");

            pTab->appendRow(pInputsGroup);
            pTab->appendRow(paramsGroup);
            pTab->appendRow(pOutputsGroup);
        }
        pRoot->appendRow(pTab);

        appendRow(pRoot);
    }
}

QString ViewParamModel::exportUI() const
{
    /*
     xml example:

     <customui>
        <node name = "VDBWrangle">
            <tab name = "Default" type="default" hint="default msg for node">
                <group name = "inputs">
                    <param name = "..." control = "..."/>
                    ...
                </group>
                <group name = "params">
                    <param name = "..." control = "..."/>
                </group>
                <group name = "outputs">
                    
                </group>
            </tab>
        </node>
        <node name = ...>
        </node>
     </customui>
    */
    return QString();
}

void ViewParamModel::onParamsInserted(const QModelIndex& parent, int first, int last)
{
    IParamModel* pModel = qobject_cast<IParamModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex& idx = pModel->index(first, 0, parent);
    if (!idx.isValid()) return;

    QStandardItem* pRoot = invisibleRootItem();
    PARAM_CLASS cls = pModel->paramClass();
    if (cls == PARAM_INPUT)
    {
        QList<QStandardItem*> lst = findItems("In Sockets", Qt::MatchRecursive | Qt::MatchExactly);
    }
    else if (cls == PARAM_PARAM)
    {

    }
    else if (cls == PARAM_OUTPUT)
    {

    }
}

void ViewParamModel::onParamsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    IParamModel* pModel = qobject_cast<IParamModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex& idx = pModel->index(first, 0, parent);
    if (!idx.isValid()) return;


}