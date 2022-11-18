#include "customuirw.h"
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/include/modelrole.h>
#include "zassert.h"
#include <zenomodel/include/uihelper.h>


namespace zenoio
{
    void exportItem(VParamItem* pItem, RAPIDJSON_WRITER& writer)
    {
        JsonObjBatch batch(writer);

        if (!pItem)
            return;

        int vType = pItem->data(ROLE_VPARAM_TYPE).toInt();
        if (vType == VPARAM_TAB)
        {
            for (int r = 0; r < pItem->rowCount(); r++)
            {
                VParamItem* pGroup = static_cast<VParamItem*>(pItem->child(r));
                ZASSERT_EXIT(pGroup && pGroup->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);
                const QString& vName = pGroup->data(ROLE_VPARAM_NAME).toString();
                writer.Key(vName.toUtf8());
                pGroup->exportJson(writer);
            }
        }
        else if (vType == VPARAM_GROUP)
        {
            for (int r = 0; r < pItem->rowCount(); r++)
            {
                VParamItem* pChild = static_cast<VParamItem*>(pItem->child(r));
                ZASSERT_EXIT(pChild && pChild->data(ROLE_VPARAM_TYPE) == VPARAM_PARAM);
                const QString& vName = pChild->data(ROLE_VPARAM_NAME).toString();
                writer.Key(vName.toUtf8());
                pChild->exportJson(writer);
            }
        }
        else if (vType == VPARAM_PARAM)
        {
            bool bCoreParam = pItem->data(ROLE_VPARAM_IS_COREPARAM).toBool();
            const QString& corename = pItem->data(ROLE_PARAM_NAME).toString();
            writer.Key("core-param");
            writer.String(corename.toUtf8());

            writer.Key("control");
            {
                JsonObjBatch _scope(writer);

                PARAM_CONTROL ctrl = (PARAM_CONTROL)pItem->data(ROLE_PARAM_CTRL).toInt();
                QString typeDesc = UiHelper::getControlDesc(ctrl);

                writer.Key("name");
                writer.String(typeDesc.toUtf8());

                if (!bCoreParam)
                {
                    writer.Key("value");
                    const QVariant& value = pItem->data(ROLE_PARAM_VALUE);
                    const QString& coreType = pItem->data(ROLE_PARAM_TYPE).toString();
                    JsonHelper::AddVariant(value, coreType, writer, true);
                }

                if (ctrl == CONTROL_ENUM)
                {
                    CONTROL_PROPERTIES pros = pItem->data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();
                    ZASSERT_EXIT(pros.find("items") != pros.end());
                    QStringList items = pros["items"].toStringList();

                    writer.Key("items");
                    writer.StartArray();
                    for (QString item : items)
                    {
                        writer.String(item.toUtf8());
                    }
                    writer.EndArray();
                }
                else if (ctrl == CONTROL_SPINBOX_SLIDER || ctrl == CONTROL_HSPINBOX ||
                    ctrl == CONTROL_HSLIDER)
                {
                    CONTROL_PROPERTIES pros = pItem->data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();

                    writer.Key("step");
                    writer.Int(pros["step"].toInt());

                    writer.Key("min");
                    writer.Int(pros["min"].toInt());

                    writer.Key("max");
                    writer.Int(pros["max"].toInt());
                }
            }
            //todo: link.
        }
    }

    void exportCustomUI(ViewParamModel* pModel, RAPIDJSON_WRITER& writer)
    {
        JsonObjBatch batch(writer);
        QStandardItem* pRoot = pModel->invisibleRootItem()->child(0);
        for (int r = 0; r < pRoot->rowCount(); r++)
        {
            VParamItem* pTab = static_cast<VParamItem*>(pRoot->child(r));
            const QString& vName = pTab->data(ROLE_VPARAM_NAME).toString();
            writer.Key(vName.toUtf8());
            exportItem(pTab, writer);
        }
    }
}
