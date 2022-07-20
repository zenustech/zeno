#include "serialize.h"
#include <model/graphsmodel.h>
#include <zeno/utils/logger.h>
#include <model/modeldata.h>
#include <model/modelrole.h>
#include <zenoui/util/uihelper.h>
#include "util/log.h"
#include "util/apphelper.h"

using namespace JsonHelper;

static QString nameMangling(const QString& prefix, const QString& ident) {
    if (prefix.isEmpty())
        return ident;
    else
        return prefix + "/" + ident;
}

static void serializeGraph(IGraphsModel* pGraphsModel, const QModelIndex& subgIdx, QString const &graphIdPrefix, bool bView, RAPIDJSON_WRITER& writer)
{
    ZASSERT_EXIT(pGraphsModel && subgIdx.isValid());

    QModelIndexList subgNodes, normNodes;
    pGraphsModel->getNodeIndices(subgIdx, subgNodes, normNodes);

    //scan all the nodes in the subgraph.
    for (int r = 0; r < pGraphsModel->itemCount(subgIdx); r++)
	{
        const QModelIndex& idx = pGraphsModel->index(r, subgIdx);
        QString ident = idx.data(ROLE_OBJID).toString();
        ident = nameMangling(graphIdPrefix, ident);
        const QString& name = idx.data(ROLE_OBJNAME).toString();

		int opts = idx.data(ROLE_OPTIONS).toInt();
        QString noOnceIdent;
        if (opts & OPT_ONCE) {
            noOnceIdent = ident;
            ident = ident + ":RUNONCE";
        }

        bool bSubgNode = subgNodes.indexOf(idx) != -1;

        const INPUT_SOCKETS &inputs = idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        const OUTPUT_SOCKETS &outputs = idx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();

        if (opts & OPT_MUTE) {
            AddStringList({ "addNode", "HelperMute", ident }, writer);
        } else {
            if (!bSubgNode) {
                AddStringList({"addNode", name, ident}, writer);
            } else {
                AddStringList({"addSubnetNode", name, ident}, writer);
                //for (INPUT_SOCKET input : inputs) {
                    //AddStringList({"addSubnetInput", ident, input.info.name}, writer);
                //}
                //for (OUTPUT_SOCKET output : outputs) {
                    //AddStringList({"addSubnetOutput", ident, output.info.name}, writer);
                //}
                AddStringList({"pushSubnetScope", ident}, writer);
                const QString& prefix = nameMangling(graphIdPrefix, idx.data(ROLE_OBJID).toString());
                bool _bView = bView && (idx.data(ROLE_OPTIONS).toInt() & OPT_VIEW);
                serializeGraph(pGraphsModel, pGraphsModel->index(name), prefix, _bView, writer);
                AddStringList({"popSubnetScope", ident}, writer);
            }
        }

        auto outputIt = outputs.begin();

        for (INPUT_SOCKET input : inputs)
        {
            auto inputName = input.info.name;

            if (opts & OPT_MUTE) {
                if (outputIt != outputs.end()) {
                    OUTPUT_SOCKET output = *outputIt++;
                    inputName = output.info.name; // HelperMute forward all inputs to outputs by socket name
                } else {
                    inputName += ":DUMMYDEP";
                }
            }

            if (input.linkIndice.isEmpty())
            {
                const QVariant& defl = input.info.defaultValue;
                if (!defl.isNull())
                {
                    AddVariantList({"setNodeInput", ident, inputName, defl}, input.info.type, writer);
                }
            }
            else
            {
                for (QPersistentModelIndex linkIdx : input.linkIndice)
                {
                    ZASSERT_EXIT(linkIdx.isValid());
                    QString outSock = linkIdx.data(ROLE_OUTSOCK).toString();
                    QString outId = linkIdx.data(ROLE_OUTNODE).toString();
                    const QModelIndex& idx_ = pGraphsModel->index(outId, subgIdx);
                    outId = nameMangling(graphIdPrefix, outId);
                    AddStringList({"bindNodeInput", ident, inputName, outId, outSock}, writer);
                }
            }
        }

        const PARAMS_INFO& params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
		for (PARAM_INFO param_info : params)
		{
            AddVariantList({ "setNodeParam", ident, param_info.name, param_info.value }, param_info.typeDesc, writer);
		}

        if (opts & OPT_ONCE) {
            AddStringList({ "addNode", "HelperOnce", noOnceIdent }, writer);
            for (OUTPUT_SOCKET output : outputs) {
                if (output.info.name == "DST") continue;
                AddStringList({ "bindNodeInput", noOnceIdent, output.info.name, ident, output.info.name }, writer);
            }
            AddStringList({ "completeNode", ident }, writer);
            ident = noOnceIdent;//must before OPT_VIEW branch
        }

        for (OUTPUT_SOCKET output : outputs) {
            //the output key of the dict has not descripted by the core, need to add it manually.
            if (output.info.control == CONTROL_DICTKEY) {
                AddStringList({"addNodeOutput", ident, output.info.name}, writer);
            }     
        }

        AddStringList({ "completeNode", ident }, writer);

		if (bView && (opts & OPT_VIEW))
        {
            if (name == "SubOutput")
            {
                auto viewerIdent = ident + ":TOVIEW";
                AddStringList({"addNode", "ToView", viewerIdent}, writer);
                AddStringList({"bindNodeInput", viewerIdent, "object", ident, "_OUT_port"}, writer);
                bool isStatic = opts & OPT_ONCE;
                AddVariantList({"setNodeInput", viewerIdent, "isStatic", isStatic}, "int", writer);
                AddStringList({"completeNode", viewerIdent}, writer);
            }
            else
            {
                for (OUTPUT_SOCKET output : outputs)
                {
                    //if (output.info.name == "DST" && outputs.size() > 1)
                        //continue;
                    auto viewerIdent = ident + ":TOVIEW";
                    AddStringList({"addNode", "ToView", viewerIdent}, writer);
                    AddStringList({"bindNodeInput", viewerIdent, "object", ident, output.info.name}, writer);
                    bool isStatic = opts & OPT_ONCE;
                    AddVariantList({"setNodeInput", viewerIdent, "isStatic", isStatic}, "int", writer);
                    AddStringList({"completeNode", viewerIdent}, writer);
                    break;  //current node is not a subgraph node, so only one output is needed to view this obj.
                }
            }
        }
	}
}

void serializeScene(IGraphsModel* pModel, RAPIDJSON_WRITER& writer)
{
    serializeGraph(pModel, pModel->index("main"), "", true, writer);
}


static void appendSerializedCharArray(QString &res, const char *buf, size_t len) {
    for (auto p = buf; p < buf + len; p++) {
        res.append(QString::number((int)(uint8_t)*p));
        res.append(',');
    }
    res.append('0');
}

QString translateGraphToCpp(const char *subgJson, size_t subgJsonLen, IGraphsModel *model)
{
    QString res = R"RAW(/* auto generated from: )RAW";
    res.append(model->filePath());
    res.append(R"RAW( */
#include <zeno/extra/ISubgraphNode.h>
#include <zeno/zeno.h>
namespace {
)RAW");

    decltype(auto) descs = model->descriptors();
    for (int i = 0; i < model->rowCount(); i++) {
        auto key = model->name(model->index(i, 0));
        if (key == "main") continue;
        if (!descs.contains(key)) {
            zeno::log_warn("cannot find subgraph `{}` in descriptors table", key.toStdString());
            continue;
        }
        auto const &desc = descs[key];

        res.append(R"RAW(
struct )RAW");
        res.append(key);
        res.append(R"RAW( final : zeno::ISerialSubgraphNode {
    static const uint8_t mydata[] = {)RAW");
        appendSerializedCharArray(res, subgJson, subgJsonLen);
        res.append(R"RAW(};

    virtual const char *get_subgraph_json() override {
        return (const char *)mydata;
    }
};

ZENO_DEFNODE()RAW");

        res.append(key);
        res.append(R"RAW()({
    {)RAW");
        for (auto const &[_, entry] : desc.inputs) {
            res.append(R"RAW({")RAW");
            res.append(entry.info.type);
            res.append(R"RAW(", ")RAW");
            res.append(entry.info.name);
            res.append(R"RAW(", ")RAW");
            res.append(UiHelper::variantToString(entry.info.defaultValue));
            res.append(R"RAW("}, )RAW");
        }
        res.append(R"RAW(},
    {)RAW");
        for (auto const &[_, entry] : desc.outputs) {
            res.append(R"RAW({")RAW");
            res.append(entry.info.type);
            res.append(R"RAW(", ")RAW");
            res.append(entry.info.name);
            res.append(R"RAW(", ")RAW");
            res.append(UiHelper::variantToString(entry.info.defaultValue));
            res.append(R"RAW("}, )RAW");
        }
        res.append(R"RAW(},
    {)RAW");
        for (auto const &entry : desc.params) {
            res.append(R"RAW({")RAW");
            res.append(entry.typeDesc);
            res.append(R"RAW(", ")RAW");
            res.append(entry.name);
            res.append(R"RAW(", ")RAW");
            res.append(UiHelper::variantToString(entry.defaultValue));
            res.append(R"RAW("}, )RAW");
        }
        res.append(R"RAW(},
    {)RAW");
        for (auto const &category : desc.categories) {
            res.append(R"RAW(")RAW");
            res.append(category);
            res.append(R"RAW(", )RAW");
        }
        res.append(R"RAW(},
});
)RAW");
    }
    res.append(R"RAW(
}
)RAW");
    return res;
}
