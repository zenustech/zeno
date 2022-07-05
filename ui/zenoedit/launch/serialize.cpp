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

static void serializeSubInput(
                    const QString& ident,
                    const QString& name,
                    const PARAMS_INFO& params,
                    const QModelIndex& subNode,
                    QString const& upperPrefix,
                    RAPIDJSON_WRITER& writer)
{
    ZASSERT_EXIT(params.find("name") != params.end());
    const QString &sockName = params["name"].value.toString();
    if (subNode.isValid())
    {
        PARAMS_INFO parentParams = subNode.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        INPUT_SOCKETS upInputs = subNode.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        ZASSERT_EXIT(upInputs.find(sockName) != upInputs.end());
        const INPUT_SOCKET &upInput = upInputs[sockName];
        if (upInput.linkIndice.isEmpty())
        {
            for (PARAM_INFO param_info : params)
            {
                if (param_info.name == "defl")
                {
                    //tothink: if subNode's defl val is omitted, should we get defl value from this SubInput.
                    QVariant defl = upInput.info.defaultValue;
                    if (!defl.isValid())
                        defl = param_info.defaultValue;
                    AddVariantList({"setNodeParam", ident, "defl", defl}, upInput.info.type, writer);
                    AddVariantList({"setNodeInput", ident, "_IN_hasValue", true}, "", writer);
                }
                else
                {
                    AddVariantList({"setNodeParam", ident, param_info.name, param_info.value}, param_info.typeDesc, writer);
                }
            }
        }
        else
        {
            for (auto linkIdx : upInput.linkIndice)
            {
                const QString &inNode = linkIdx.data(ROLE_INNODE).toString();
                const QString &inSock = linkIdx.data(ROLE_INSOCK).toString();
                QString outNode = linkIdx.data(ROLE_OUTNODE).toString();
                const QString &outSock = linkIdx.data(ROLE_OUTSOCK).toString();
                ZASSERT_EXIT(inSock == sockName);

                outNode = nameMangling(upperPrefix, outNode);
                AddStringList({"bindNodeInput", ident, "_IN_port", outNode, outSock}, writer);
                AddVariantList({"setNodeInput", ident, "_IN_hasValue", true}, "", writer);
            }
        }
    }
}

static void serializeInputs(
                IGraphsModel* pGraphsModel,
                const QModelIndex &subgIdx,
                const QString& ident,
                const QModelIndex& nodeIdx,
                QString const &graphIdPrefix,
                RAPIDJSON_WRITER& writer)
{
    const int opts = nodeIdx.data(ROLE_OPTIONS).toInt();

    const OUTPUT_SOCKETS& outputs = nodeIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    auto outputIt = outputs.begin();

    const INPUT_SOCKETS& inputs = nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
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
            const QVariant &defl = input.info.defaultValue;
            if (!defl.isNull()) {
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

                //the node of outId may be a subgNode, if so, reconnect into "SubOutput".
                if (pGraphsModel->IsSubGraphNode(idx_))
                {
                    const QString &subgName = idx_.data(ROLE_OBJNAME).toString();
                    const QModelIndex &subnodeIdx =
                        AppHelper::getSubInOutNode(pGraphsModel, pGraphsModel->index(subgName), outSock, false);
                    if (subnodeIdx.isValid())
                    {
                        outSock = "_OUT_port";
                        const QString &ident_ = nameMangling(graphIdPrefix, idx_.data(ROLE_OBJID).toString());
                        outId = nameMangling(ident_, subnodeIdx.data(ROLE_OBJID).toString());
                    }
                }

                AddStringList({"bindNodeInput", ident, inputName, outId, outSock}, writer);
            }
        }
    }
}


static void serializeGraph(
                IGraphsModel* pGraphsModel,
                const QModelIndex& subgIdx,     //current subgraph idx, refers to a graph.
                const QModelIndex& subNode,     //the node of subgraph on the upper layer, refers to a node.
                QString const &graphIdPrefix,   //.
                QString const &upperPrefix,     //..
                bool bView,
                RAPIDJSON_WRITER& writer)
{
    ZASSERT_EXIT(pGraphsModel && subgIdx.isValid());

    //scan all the nodes in the subgraph.
    for (int r = 0; r < pGraphsModel->itemCount(subgIdx); r++)
	{
        const QModelIndex& idx = pGraphsModel->index(r, subgIdx);
        int opts = idx.data(ROLE_OPTIONS).toInt();
        const QString& name = idx.data(ROLE_OBJNAME).toString();
        QString ident = idx.data(ROLE_OBJID).toString();
        ident = nameMangling(graphIdPrefix, ident);

        if (pGraphsModel->IsSubGraphNode(idx))
        {
            //now, we want to expand the subgraph node recursively.
            //so, we need to serialize the subgraph first, and then build the connection with other nodes in this graph.
            const QString &prefix = ident;
            bool _bView = bView && (opts & OPT_VIEW);
            serializeGraph(pGraphsModel, pGraphsModel->index(name), idx, prefix, graphIdPrefix, _bView, writer);
            continue;
        }

        QString noOnceIdent;
        if (opts & OPT_ONCE) {
            noOnceIdent = ident;
            ident = ident + ":RUNONCE";
        }

        if (opts & OPT_MUTE) {
            AddStringList({ "addNode", "HelperMute", ident }, writer);
        } else {
            if (!pGraphsModel->index(name).isValid()) {
                AddStringList({"addNode", name, ident}, writer);
            }
        }

        const OUTPUT_SOCKETS& outputs = idx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        auto outputIt = outputs.begin();

        const INPUT_SOCKETS& inputs = idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        const PARAMS_INFO& params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();

        if (name == "SubInput")
        {
            serializeSubInput(ident, name, params, subNode, upperPrefix, writer);
        }
        else
        {
            serializeInputs(pGraphsModel, subgIdx, ident, idx, graphIdPrefix, writer);
            for (PARAM_INFO param_info : params)
            {
                AddVariantList({"setNodeParam", ident, param_info.name, param_info.value}, param_info.typeDesc, writer);
            }
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
            for (OUTPUT_SOCKET output : outputs)
            {
                if (output.info.name == "DST" /*&& outputs.size() > 1*/)
                    continue;
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

void serializeScene(IGraphsModel* pModel, RAPIDJSON_WRITER& writer)
{
    serializeGraph(pModel, pModel->index("main"), QModelIndex(), "", "", true, writer);
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
