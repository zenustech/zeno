#include "customuirw.h"
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/include/modelrole.h>
#include "zassert.h"
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/modeldata.h>
#include <rapidjson/document.h>


namespace zenomodel
{
    void exportItem(const VParamItem* pItem, RAPIDJSON_WRITER& writer)
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
                exportItem(pGroup, writer);
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
                exportItem(pChild, writer);
            }
        }
        else if (vType == VPARAM_PARAM)
        {
            bool bCoreParam = pItem->data(ROLE_VPARAM_IS_COREPARAM).toBool();
            const QString& corename = pItem->data(ROLE_PARAM_NAME).toString();
            if (!corename.isEmpty())
            {
                writer.Key("core-param");
                JsonObjBatch _scope(writer);

                writer.Key("name");
                writer.String(corename.toUtf8());

                writer.Key("class");
                PARAM_CLASS cls = (PARAM_CLASS)pItem->data(ROLE_PARAM_CLASS).toInt();
                switch (cls)
                {
                case PARAM_INPUT:
                    writer.String("input");
                    break;
                case PARAM_PARAM:
                    writer.String("parameter");
                    break;
                case PARAM_OUTPUT:
                    writer.String("output");
                    break;
                default:
                    writer.String("");
                    break;
                }
            }

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
                    writer.Key("items");
                    writer.StartArray();
                    if (pros.find("items") != pros.end())
                    {
                        QStringList items = pros["items"].toStringList();
                        for (QString item : items)
                        {
                            writer.String(item.toUtf8());
                        }
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

    VPARAM_INFO importParam(const QString& paramName, const rapidjson::Value& paramVal)
    {
        VPARAM_INFO param;
        param.vType = VPARAM_PARAM;

        if (paramVal.HasMember("core-param"))
        {
            const rapidjson::Value& coreParam = paramVal["core-param"];
            ZASSERT_EXIT(coreParam.HasMember("name") && coreParam.HasMember("class"), param);

            param.coreParam = QString::fromLocal8Bit(coreParam["name"].GetString());
            const QString& cls = QString::fromLocal8Bit(coreParam["class"].GetString());

            if (cls == "input")
            {
                param.m_cls = PARAM_INPUT;
            }
            else if (cls == "parameter")
            {
                param.m_cls = PARAM_PARAM;
            }
            else if (cls == "output")
            {
                param.m_cls = PARAM_OUTPUT;
            }
            else
            {
                param.m_cls = PARAM_UNKNOWN;
            }
        }

        ZASSERT_EXIT(paramVal.HasMember("control"), param);
        const rapidjson::Value& controlObj = paramVal["control"];

        const QString& ctrlName = QString::fromLocal8Bit(controlObj["name"].GetString());
        param.m_info.control = UiHelper::getControlByDesc(ctrlName);
        param.m_info.typeDesc = UiHelper::getTypeByControl(param.m_info.control);
        param.m_info.name = paramName;

        if (controlObj.HasMember("value"))
        {
            param.m_info.value = UiHelper::parseJson(controlObj["value"], nullptr);
        }
        if (controlObj.HasMember("items"))
        {
            //combobox
            ZASSERT_EXIT(controlObj["items"].IsArray(), param);
            QStringList lstItems = UiHelper::parseJson(controlObj["items"]).toStringList();
            param.controlInfos = lstItems;
        }
        if (controlObj.HasMember("step") && controlObj.HasMember("min") && controlObj.HasMember("max"))
        {
            int step = controlObj["step"].GetInt();
            int min = controlObj["min"].GetInt();
            int max = controlObj["max"].GetInt();
            SLIDER_INFO sliderInfo;
            sliderInfo.max = max;
            sliderInfo.min = min;
            sliderInfo.step = step;
            param.controlInfos = QVariant::fromValue(sliderInfo);
        }

        return param;
    }

    VPARAM_INFO importCustomUI(const rapidjson::Value& jsonCutomUI)
    {
        VPARAM_INFO invisibleRoot;
        for (const auto& tabObj : jsonCutomUI.GetObject())
        {
            const QString& tabName = tabObj.name.GetString();
            VPARAM_INFO tab;
            tab.vType = VPARAM_TAB;
            tab.m_info.name = tabName;

            for (const auto& groupObj : tabObj.value.GetObject())
            {
                const QString& groupName = groupObj.name.GetString();
                VPARAM_INFO group;
                group.vType = VPARAM_GROUP;
                group.m_info.name = groupName;

                for (const auto& paramObj : groupObj.value.GetObject())
                {
                    const QString& paramName = paramObj.name.GetString();
                    const rapidjson::Value& paramVal = paramObj.value;

                    VPARAM_INFO param = importParam(paramName, paramVal);
                    group.children.append(param);
                }

                tab.children.append(group);
            }

            invisibleRoot.children.append(tab);
        }
        return invisibleRoot;
    }
}
