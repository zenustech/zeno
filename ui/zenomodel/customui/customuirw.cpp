#include "customuirw.h"
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/include/vparamitem.h>
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
            if (pItem->m_index.isValid())
            {
                writer.Key("core-param");
                JsonObjBatch _scope(writer);

                const QString& refPath = pItem->m_index.data(ROLE_OBJPATH).toString();

                writer.Key("name");
                writer.String(refPath.toUtf8());

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

            QVariant deflVal = pItem->data(ROLE_PARAM_VALUE);
            const QString &type = pItem->data(ROLE_PARAM_TYPE).toString();
            bool bValid = UiHelper::validateVariant(deflVal, type);
            if (bValid && !pItem->m_index.isValid()) {
                writer.Key("value");
                JsonHelper::AddVariant(deflVal, type, writer, true);
            }

            PARAM_CONTROL ctrl = (PARAM_CONTROL)pItem->data(ROLE_PARAM_CTRL).toInt();
            CONTROL_PROPERTIES pros = pItem->data(ROLE_VPARAM_CTRL_PROPERTIES).value<CONTROL_PROPERTIES>();

            writer.Key("control");
            JsonHelper::dumpControl(ctrl, pros, writer);

            //disable drag!!!
            writer.Key("uuid");
            writer.Uint(pItem->m_uuid);
            //todo: link.

            if (pItem->m_customData.contains(ROLE_VPARAM_TOOLTIP)
                && !pItem->m_customData[ROLE_VPARAM_TOOLTIP].toString().isEmpty())
            {
                writer.Key("tooltip");
                writer.String(pItem->m_customData[ROLE_VPARAM_TOOLTIP].toString().toUtf8());
            }
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

            param.refParamPath = QString::fromUtf8(coreParam["name"].GetString());
            const QString& cls = QString::fromUtf8(coreParam["class"].GetString());

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

        if (paramVal.HasMember("uuid"))
        {
            const rapidjson::Value& uuidObj = paramVal["uuid"];
            ZASSERT_EXIT(uuidObj.IsUint(), param);
            param.m_uuid = uuidObj.GetUint();
        }

        ZASSERT_EXIT(paramVal.HasMember("control"), param);
        const rapidjson::Value& controlObj = paramVal["control"];

        if (controlObj.HasMember("items") || (controlObj.HasMember("step") && controlObj.HasMember("min") && controlObj.HasMember("max")))
        {
            JsonHelper::importControl(controlObj, param.m_info.control, param.controlInfos);
        } 
        else 
        {
            const QString &ctrlName = QString::fromUtf8(controlObj["name"].GetString());
            param.m_info.control = UiHelper::getControlByDesc(ctrlName);
        }
        param.m_info.typeDesc = UiHelper::getTypeByControl(param.m_info.control);
        param.m_info.name = paramName;

        if (paramVal.HasMember("value"))
        {
            param.m_info.value = UiHelper::parseJson(paramVal["value"], nullptr);
        }

        if (paramVal.HasMember("tooltip")) 
        {
            param.m_info.toolTip = QString::fromUtf8(paramVal["tooltip"].GetString());
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
