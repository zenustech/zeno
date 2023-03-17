#include "zsgwriter.h"
#include <zenomodel/include/modelrole.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zenomodel/include/uihelper.h>
#include "variantptr.h"
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/customui/customuirw.h>

using namespace zeno::iotags;


ZsgWriter::ZsgWriter()
{
}

ZsgWriter& ZsgWriter::getInstance()
{
    static ZsgWriter writer;
    return writer;
}

void ZsgWriter::dumpToClipboard(const QMap<QString, NODE_DATA>& nodes)
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

QString ZsgWriter::dumpProgramStr(IGraphsModel* pModel, APP_SETTINGS settings)
{
    QString strJson;
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

        writer.Key("views");
        {
            writer.StartObject();
            dumpTimeline(settings.timeline, writer);
            writer.EndObject();
        }

        NODE_DESCS descs = pModel->descriptors();
        writer.Key("descs");
        _dumpDescriptors(descs, writer);

        writer.Key("version");
        writer.String("v2.5");  //distinguish the new version ui from the stable zeno2.
    }

    strJson = QString::fromUtf8(s.GetString());
    return strJson;
}

void ZsgWriter::_dumpSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx, RAPIDJSON_WRITER& writer)
{
    JsonObjBatch batch(writer);
    if (!pModel)
        return;

    {
        writer.Key("nodes");
        JsonObjBatch _batch(writer);

        int n = pModel->itemCount(subgIdx);
        for (int i = 0; i < n; i++)
        {
            const QModelIndex& idx = pModel->index(i, subgIdx);
            const NODE_DATA& node = pModel->itemData(idx, subgIdx);
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

void ZsgWriter::dumpSocket(SOCKET_INFO socket, bool bInput, RAPIDJSON_WRITER& writer, bool bDesc)
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
            } else if (socket.sockProp & SOCKPROP_EDITABLE) {
                writer.String("editable");
            } else if (socket.sockProp & SOCKPROP_GROUP_LINE) {
                writer.String("group-line");
            } else {
                writer.String("normal");
            }
        }
    }

    if (socket.sockProp == SOCKPROP_DICTLIST_PANEL)
    {
        //dict param keys
        writer.Key("dictlist-panel");
        writer.StartObject();
        {
            writer.Key("collasped");
            writer.Bool(socket.dictpanel.bCollasped);

            writer.Key("keys");
            writer.StartObject();
            for (int i = 0; i < socket.dictpanel.keys.size(); i++)
            {
                const DICTKEY_INFO &info = socket.dictpanel.keys[i];
                writer.Key(info.key.toUtf8());
                writer.StartObject();
                if (bInput) //no need to export link on output key sockets.
                {
                    writer.Key("link");
                    QString otherLinkSock = info.links.isEmpty() ? "" : info.links[0].outSockPath;
                    if (otherLinkSock.isEmpty())
                        writer.Null();
                    else
                        writer.String(otherLinkSock.toUtf8());
                }
                writer.EndObject();
            }
            writer.EndObject();
        }
        writer.EndObject();
    }

    if (bInput && !bDesc)
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

        bool bValid = UiHelper::validateVariant(deflVal, sockType);
        if (!bValid)
            deflVal = QVariant();
        AddVariant(deflVal, sockType, writer, true);

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

void ZsgWriter::dumpNode(const NODE_DATA& data, RAPIDJSON_WRITER& writer)
{
    JsonObjBatch batch(writer);

    writer.Key("name");
    const QString& name = data[ROLE_OBJNAME].toString();
    writer.String(name.toUtf8());

    const INPUT_SOCKETS& inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    const OUTPUT_SOCKETS& outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
    const PARAMS_INFO& params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();

    writer.Key("inputs");
    {
        JsonObjBatch _batch(writer);
        for (const INPUT_SOCKET& inSock : inputs)
        {
            writer.Key(inSock.info.name.toUtf8());

#ifndef LEGACY_ZSG_WRITE
            //adapt new io format, bacause the legacy io format is very restricted!
            dumpSocket(inSock.info, true, writer, false);
#else
            QVariant deflVal = inSock.info.defaultValue;
            const QString& sockType = inSock.info.type;
            bool bValid = UiHelper::validateVariant(deflVal, sockType);
            if (!inSock.info.links.isEmpty())
            {
                //legacy link.
                const EdgeInfo& link = inSock.info.links[0];
                QString outputNode = UiHelper::getSockNode(link.outSockPath);
                QString outputSock = UiHelper::getSockName(link.outSockPath);
                AddVariantList({ outputNode, outputSock, deflVal }, sockType, writer, true);
            }
            else
            {
                if (!bValid)
                    deflVal = QVariant();
                AddVariantList({ QVariant(), QVariant(), deflVal }, sockType, writer, true);
            }
#endif
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
        for (const OUTPUT_SOCKET &outSock : outputs)
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

void ZsgWriter::dumpTimeline(TIMELINE_INFO info, RAPIDJSON_WRITER& writer)
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
    }
}

void ZsgWriter::dumpParams(const PARAM_INFO &info, RAPIDJSON_WRITER &writer) 
{
    writer.StartObject();

    writer.Key("value");
    AddVariant(info.value, info.typeDesc, writer, true);

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

void ZsgWriter::dumpSubGraphDesc(const NODE_DESC &desc, RAPIDJSON_WRITER& writer) 
{
    //inputs
    writer.Key("inputs");
    writer.StartObject();
    for (INPUT_SOCKET inSock : desc.inputs) {
        writer.Key(inSock.info.name.toUtf8());
        dumpSocket(inSock.info, true, writer, true);;
    }
    writer.EndObject();
    //params
    writer.Key("params");
    writer.StartObject();
    for (PARAM_INFO param : desc.params) {
        writer.Key(param.name.toUtf8());
        dumpParams(param, writer);
    }
    writer.EndObject();
    //outputs
    writer.Key("outputs");
    writer.StartObject();
    for (OUTPUT_SOCKET outSock : desc.outputs) {
        writer.Key(outSock.info.name.toUtf8());
        dumpSocket(outSock.info, false, writer, true);
    }
    writer.EndObject();

    writer.Key("categories");
    AddStringList(desc.categories, writer);

    writer.Key("is_subgraph");
    writer.Bool(true);
}

void ZsgWriter::_dumpDescriptors(const NODE_DESCS& descs, RAPIDJSON_WRITER& writer)
{
    JsonObjBatch batch(writer);

    for (auto name : descs.keys())
    {
        const NODE_DESC& desc = descs[name];

        writer.Key(name.toUtf8());
        JsonObjBatch _batch(writer);

        if(desc.is_subgraph) 
        {
            dumpSubGraphDesc(desc, writer);
        } 
        else 
        {
            {
                writer.Key("inputs");
                JsonArrayBatch _batchArr(writer);

                for (INPUT_SOCKET inSock : desc.inputs)
                {
                    AddVariantToStringList({inSock.info.type, inSock.info.name, inSock.info.defaultValue}, writer);
                }
            }
            {
                writer.Key("params");
                JsonArrayBatch _batchArr(writer);

                for (PARAM_INFO param : desc.params)
                {
                    AddVariantToStringList({param.typeDesc, param.name, param.defaultValue}, writer);
                }
            }

            {
                writer.Key("outputs");
                JsonArrayBatch _batchArr(writer);

                for (OUTPUT_SOCKET outSock : desc.outputs)
                {
                    AddVariantToStringList({outSock.info.type, outSock.info.name, outSock.info.defaultValue}, writer);
                }
            }

            writer.Key("categories");
            AddStringList(desc.categories, writer);
        }
    }
}
