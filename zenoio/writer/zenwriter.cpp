#include "ZenWriter.h"
#include <zenomodel/include/modelrole.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zenomodel/include/uihelper.h>
#include "variantptr.h"
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/customui/customuirw.h>

using namespace zeno::iotags;


namespace zenoio
{
    ZenWriter::ZenWriter()
    {
    }

    void ZenWriter::dumpToClipboard(const GraphData& nodes)
    {
        QString strJson;

        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);
        {
            JsonObjBatch batch(writer);
            writer.Key("nodes");
            {
                JsonObjBatch _batch(writer);
                for (const QString& nodeId : nodes.keys())
                {
                    const NODE_DATA& nodeData = nodes[nodeId];
                    if (nodeData.find(ROLE_NODETYPE) != nodeData.end() &&
                        NO_VERSION_NODE == nodeData[ROLE_NODETYPE])
                    {
                        continue;
                    }

                    writer.Key(nodeId.toUtf8());
                    dumpNode(nodeData, writer);
                }
            }
        }

        strJson = QString::fromUtf8(s.GetString());
        QMimeData* pMimeData = new QMimeData;
        pMimeData->setText(strJson);
        QApplication::clipboard()->setMimeData(pMimeData);
    }

    std::string ZenWriter::dumpProgramStr(zeno::GraphData maingraph, AppSettings settings)
    {
        std::string strJson;
        if (!pModel)
            return strJson;

        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);

        {
            JsonObjBatch batch(writer);

            writer.Key("graph");
            {
                JsonObjBatch _batch(writer);
                for (int i = 0; i < pModel->rowCount(); i++)
                {
                    const QModelIndex& subgIdx = pModel->index(i, 0);
                    const QString& subgName = subgIdx.data(ROLE_OBJNAME).toString();
                    writer.Key(subgName.toUtf8());
                    _dumpSubGraph(pModel, subgIdx, writer);
                }
            }

            const FuckQMap<QString, CommandParam>& commandParams = pModel->commandParams();
            if (!commandParams.isEmpty())
            {
                writer.Key("command");
                {
                    writer.StartObject();
                    for (const auto& key : commandParams.keys())
                    {
                        writer.Key(key.toUtf8());
                        writer.StartObject();
                        writer.Key("name");
                        writer.String(commandParams[key].name.toUtf8());
                        writer.Key("description");
                        writer.String(commandParams[key].description.toUtf8());
                        writer.EndObject();
                    }
                    writer.EndObject();
                }
            }

            writer.Key("views");
            {
                writer.StartObject();
                dumpTimeline(settings.timeline, writer);
                writer.EndObject();
            }

            writer.Key("settings");
            dumpSettings(settings, writer);

            writer.Key("version");
            writer.String("v2.5");  //distinguish the new version ui from the stable zeno2.
        }

        strJson = QString::fromUtf8(s.GetString());
        return strJson;
    }

    void ZenWriter::_dumpSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx, RAPIDJSON_WRITER& writer)
    {
        JsonObjBatch batch(writer);
        if (!pModel)
            return;

        {
            int type = subgIdx.data(ROLE_SUBGRAPH_TYPE).toInt();
            writer.Key("type");
            writer.Int(type);
            writer.Key("nodes");
            JsonObjBatch _batch(writer);

            int n = pModel->itemCount(subgIdx);
            for (int i = 0; i < n; i++)
            {
                const QModelIndex& idx = pModel->index(i, subgIdx);
                const NODE_DATA& node = pModel->itemData(idx, subgIdx);
                if (node.find(ROLE_NODETYPE) != node.end() && NO_VERSION_NODE == node[ROLE_NODETYPE])
                {
                    continue;
                }
                const QString& id = node[ROLE_OBJID].toString();
                writer.Key(id.toUtf8());
                dumpNode(node, writer);
            }
        }
        {
            writer.Key("view_rect");
            JsonObjBatch _batch(writer);
            QRectF viewRc = pModel->viewRect(subgIdx);
            if (!viewRc.isNull())
            {
                writer.Key("x"); writer.Double(viewRc.x());
                writer.Key("y"); writer.Double(viewRc.y());
                writer.Key("width"); writer.Double(viewRc.width());
                writer.Key("height"); writer.Double(viewRc.height());
            }
        }
    }

    void ZenWriter::dumpSocket(zeno::ParamInfo param, RAPIDJSON_WRITER& writer)
    {
        //new io format for socket.
        writer.StartObject();

        //property
        if (socket.sockProp != SOCKPROP_NORMAL)
        {
            writer.Key("property");
            {
                if (socket.sockProp & SOCKPROP_DICTLIST_PANEL) {
                    writer.String("dict-panel");
                }
                else if (socket.sockProp & SOCKPROP_EDITABLE) {
                    writer.String("editable");
                }
                else if (socket.sockProp & SOCKPROP_GROUP_LINE) {
                    writer.String("group-line");
                }
                else {
                    writer.String("normal");
                }
            }
        }

        if (bInput)
        {
            writer.Key("link");
            if (socket.links.isEmpty())
            {
                writer.Null();
            }
            else
            {
                //writer obj path directly.
                QString otherLinkSock = bInput ? socket.links[0].outSockPath : socket.links[0].inSockPath;
                writer.String(otherLinkSock.toUtf8());
            }
        }

        const QString& sockType = socket.type;
        writer.Key("type");
        writer.String(sockType.toUtf8());

        if (bInput)
        {
            writer.Key("default-value");
            QVariant deflVal = socket.defaultValue;
            bool bOK = false;
            if (deflVal.canConvert<CURVES_DATA>() && (sockType == "float" || sockType.startsWith("vec"))) {
                bOK = AddVariant(deflVal, "curve", writer);
            }
            else {
                bool bValid = UiHelper::validateVariant(deflVal, sockType);
                if (!bValid)
                    deflVal = QVariant();
                bOK = AddVariant(deflVal, sockType, writer);
            }

            if (!bOK)
            {
                zeno::log_error("write default-value error. nodeId : {}, socket : {}", socket.nodeid.toStdString(), socket.name.toStdString());
            }

            writer.Key("control");
            JsonHelper::dumpControl(socket.control, socket.ctrlProps, writer);
        }

        if (!socket.toolTip.isEmpty())
        {
            writer.Key("tooltip");
            writer.String(socket.toolTip.toUtf8());
        }
        writer.EndObject();
    }

    void ZenWriter::dumpNode(const NodeData& data, RAPIDJSON_WRITER& writer)
    {
        JsonObjBatch batch(writer);

        writer.Key("name");
        const QString& name = data[ROLE_OBJNAME].toString();
        writer.String(name.toUtf8());

        const QString& customName = data[ROLE_CUSTOM_OBJNAME].toString();
        if (!customName.isEmpty())
        {
            writer.Key("customName");
            writer.String(customName.toUtf8());
        }

        const INPUT_SOCKETS& inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
        const OUTPUT_SOCKETS& outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        const PARAMS_INFO& params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();

        writer.Key("inputs");
        {
            JsonObjBatch _batch(writer);
            for (const INPUT_SOCKET& inSock : inputs)
            {
                writer.Key(inSock.info.name.toUtf8());
                dumpSocket(inSock.info, true, writer, false);
            }
        }
        writer.Key("params");
        {
            JsonObjBatch _batch(writer);
            for (const PARAM_INFO& info : params)
            {
                writer.Key(info.name.toUtf8());
                dumpParams(info, writer);
            }
        }
        writer.Key("outputs");
        {
            JsonObjBatch _scope(writer);
            for (const OUTPUT_SOCKET& outSock : outputs)
            {
                writer.Key(outSock.info.name.toUtf8());
                dumpSocket(outSock.info, false, writer, false);
            }
        }
        writer.Key("uipos");
        {
            QPointF pos = data[ROLE_OBJPOS].toPointF();
            writer.StartArray();
            writer.Double(pos.x());
            writer.Double(pos.y());
            writer.EndArray();
        }

        writer.Key("options");
        {
            QStringList options;
            int opts = data[ROLE_OPTIONS].toInt();
            if (opts & OPT_ONCE) {
                options.push_back("ONCE");
            }
            if (opts & OPT_MUTE) {
                options.push_back("MUTE");
            }
            if (opts & OPT_PREP) {
                options.push_back("PREP");
            }
            if (opts & OPT_VIEW) {
                options.push_back("VIEW");
            }
            if (data[ROLE_COLLASPED].toBool())
            {
                options.push_back("collapsed");
            }
            AddStringList(options, writer);
        }

        //dump custom keys in dictnode.
        {
            QStringList inDictKeys, outDictKeys;
            for (const INPUT_SOCKET& inSock : inputs) {
                if (inSock.info.sockProp & SOCKPROP_EDITABLE) {
                    inDictKeys.append(inSock.info.name);
                }
            }
            for (const OUTPUT_SOCKET& outSock : outputs) {
                if (outSock.info.sockProp & SOCKPROP_EDITABLE) {
                    outDictKeys.append(outSock.info.name);
                }
            }
            //replace socket_keys with dict_keys, which is more expressive.
            if (!inDictKeys.isEmpty() || !outDictKeys.isEmpty())
            {
                writer.Key("dict_keys");
                JsonObjBatch _batch(writer);

                writer.Key("inputs");
                writer.StartArray();
                for (auto inSock : inDictKeys) {
                    writer.String(inSock.toUtf8());
                }
                writer.EndArray();

                writer.Key("outputs");
                writer.StartArray();
                for (auto outSock : outDictKeys) {
                    writer.String(outSock.toUtf8());
                }
                writer.EndArray();
            }
        }

        if (name == "Blackboard") {
            // do not compatible with zeno1
            PARAMS_INFO params = data[ROLE_PARAMS_NO_DESC].value<PARAMS_INFO>();
            BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
            writer.Key("blackboard");
            {
                JsonObjBatch _batch(writer);
                writer.Key("special");
                writer.Bool(info.special);
                writer.Key("width");
                writer.Double(info.sz.width());
                writer.Key("height");
                writer.Double(info.sz.height());
                writer.Key("title");
                writer.String(info.title.toUtf8());
                writer.Key("content");
                writer.String(info.content.toUtf8());
            }

        }
        else if (name == "Group") {
            // do not compatible with zeno1
            PARAMS_INFO params = data[ROLE_PARAMS_NO_DESC].value<PARAMS_INFO>();
            BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
            writer.Key("blackboard");
            {
                JsonObjBatch _batch(writer);
                writer.Key("width");
                writer.Double(info.sz.width());
                writer.Key("height");
                writer.Double(info.sz.height());
                writer.Key("title");
                writer.String(info.title.toUtf8());
                writer.Key("background");
                writer.String(info.background.name().toUtf8());
                writer.Key("items");
                writer.StartArray();
                for (auto item : info.items) {
                    writer.String(item.toUtf8());
                }
                writer.EndArray();
            }
        }
        //custom ui for panel
        ViewParamModel* viewParams = QVariantPtr<ViewParamModel>::asPtr(data[ROLE_PANEL_PARAMS]);
        if (viewParams && viewParams->isDirty())
        {
            writer.Key("customui-panel");
            zenomodel::exportCustomUI(viewParams, writer);
        }

        //custom ui for node
        ViewParamModel* viewNodeParams = QVariantPtr<ViewParamModel>::asPtr(data[ROLE_NODE_PARAMS]);
        if (viewNodeParams && viewNodeParams->isDirty())
        {
            writer.Key("customui-node");
            zenomodel::exportCustomUI(viewNodeParams, writer);
        }
    }

    void ZenWriter::dumpTimeline(zeno::TimelineInfo info, RAPIDJSON_WRITER& writer)
    {
        writer.Key("timeline");
        {
            JsonObjBatch _batch(writer);
            writer.Key(timeline::start_frame);
            writer.Int(info.beginFrame);
            writer.Key(timeline::end_frame);
            writer.Int(info.endFrame);
            writer.Key(timeline::curr_frame);
            writer.Int(info.currFrame);
            writer.Key(timeline::always);
            writer.Bool(info.bAlways);
            writer.Key(timeline::timeline_fps);
            writer.Int(info.timelinefps);
        }
    }

    void ZenWriter::dumpParams(const PARAM_INFO& info, RAPIDJSON_WRITER& writer)
    {
        writer.StartObject();

        writer.Key("value");
        AddVariant(info.value, info.typeDesc, writer);

        writer.Key("control");
        JsonHelper::dumpControl(info.control, info.controlProps, writer);

        writer.Key("type");
        writer.String(info.typeDesc.toUtf8());

        if (!info.toolTip.isEmpty())
        {
            writer.Key("tooltip");
            writer.String(info.toolTip.toUtf8());
        }
        writer.EndObject();
    }

    void ZenWriter::dumpSettings(const APP_SETTINGS settings, RAPIDJSON_WRITER& writer)
    {
        const RECORD_SETTING& info = settings.recordInfo;
        JsonObjBatch batch(writer);
        {
            writer.Key("recordinfo");
            writer.StartObject();
            writer.Key(recordinfo::record_path);
            writer.String(info.record_path.toUtf8());
            writer.Key(recordinfo::videoname);
            writer.String(info.videoname.toUtf8());
            writer.Key(recordinfo::fps);
            writer.Int(info.fps);
            writer.Key(recordinfo::bitrate);
            writer.Int(info.bitrate);
            writer.Key(recordinfo::numMSAA);
            writer.Int(info.numMSAA);
            writer.Key(recordinfo::numOptix);
            writer.Int(info.numOptix);
            writer.Key(recordinfo::width);
            writer.Int(info.width);
            writer.Key(recordinfo::height);
            writer.Int(info.height);
            writer.Key(recordinfo::bExportVideo);
            writer.Bool(info.bExportVideo);
            writer.Key(recordinfo::needDenoise);
            writer.Bool(info.needDenoise);
            writer.Key(recordinfo::bAutoRemoveCache);
            writer.Bool(info.bAutoRemoveCache);
            writer.Key(recordinfo::bAov);
            writer.Bool(info.bAov);
            writer.Key(recordinfo::bExr);
            writer.Bool(info.bExr);
            writer.EndObject();

            writer.Key("userdatainfo");
            writer.StartObject();
            writer.Key(userdatainfo::optixShowBackground);
            writer.Bool(settings.userdataInfo.optix_show_background);
            writer.EndObject();
        }
    }
}