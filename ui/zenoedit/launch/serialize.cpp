#include "serialize.h"
#include <model/graphsmodel.h>
#include <zeno/utils/logger.h>
#include <model/modeldata.h>
#include <model/modelrole.h>
#include <zenoui/util/uihelper.h>

using namespace JsonHelper;

static void serializeGraph(SubGraphModel* pModel, GraphsModel* pGraphsModel, QStringList const &graphNames, RAPIDJSON_WRITER& writer, QString const &graphIdPrefix)
{
	const QString& name = pModel->name();
	const NODES_DATA& nodes = pModel->nodes();

	for (const NODE_DATA& node : nodes)
	{
		QString ident = node[ROLE_OBJID].toString();
        ident = graphIdPrefix + ident;
		QString name = node[ROLE_OBJNAME].toString();
        //zeno::log_critical("got node {} {}", name.toStdString(), ident.toStdString());
		const INPUT_SOCKETS& inputs = node[ROLE_INPUTS].value<INPUT_SOCKETS>();
        const OUTPUT_SOCKETS& outputs = node[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
		PARAMS_INFO params = node[ROLE_PARAMETERS].value<PARAMS_INFO>();
		int opts = node[ROLE_OPTIONS].toInt();
		/*QStringList options;
		if (opts & OPT_ONCE)
		{
			options.push_back("ONCE");
		}
		if (opts & OPT_MUTE)
		{
			options.push_back("MUTE");
		}
		if (opts & OPT_PREP)
		{
			options.push_back("PREP");
		}
		if (opts & OPT_VIEW)
		{
			options.push_back("VIEW");
		}*/
        QString noOnceIdent;
        if (opts & OPT_ONCE) {
            noOnceIdent = ident;
            ident = ident + ".RUNONCE";
        }

        if (opts & OPT_MUTE) {
            AddStringList({ "addNode", "HelperMute", ident }, writer);
        } else {

            if (graphNames.indexOf(name) != -1)
            {
                zeno::log_critical("got subgraph {}", name.toStdString());
                auto nextGraphIdPrefix = ident + "/";
                SubGraphModel* pSubModel = pGraphsModel->subGraph(name);
                serializeGraph(pSubModel, pGraphsModel, graphNames, writer, nextGraphIdPrefix);
                //ret.push_back(QJsonArray({ "pushSubgraph", ident, name }));
                //serializeGraph(pSubModel, pGraphsModel, graphNames, ret);
                //ret.push_back(QJsonArray({ "popSubgraph", ident, name }));

            } else {
                AddStringList({ "addNode", name, ident }, writer);
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
                    inputName += ".DUMMYDEP";
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
                    Q_ASSERT(linkIdx.isValid());
                    const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();
                    const QString& outId = linkIdx.data(ROLE_OUTNODE).toString();
                    AddStringList({ "bindNodeInput", ident, inputName, outId, outSock }, writer);
                }
            }
        }

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

        AddStringList({ "completeNode", ident }, writer);

		if (opts & OPT_VIEW) {
            for (OUTPUT_SOCKET output : outputs)
            {
                //if (output.info.name == "DST") continue;//qmap wants to put DST/SRC as first socket, skip it
                auto viewerIdent = ident + ".TOVIEW";
                AddStringList({"addNode", "ToView", viewerIdent}, writer);
                AddStringList({"bindNodeInput", viewerIdent, "object", ident, output.info.name}, writer);
                AddStringList({"completeNode", viewerIdent}, writer);
                break;  //???
            }
        }

		/*if (opts & OPT_MUTE) {
            auto inputIt = inputs.begin();
            for (OUTPUT_SOCKET output : outputs)
            {
                if (inputIt == inputs.end()) break;
                INPUT_SOCKET input = *++inputIt;
                input.info.name
            }
        }*/

        // mock options at editor side, done
		/*for (QString optionName : options)
		{
			ret.push_back(QJsonArray({"setNodeOption", ident, optionName}));
		}*/
	}
}

void serializeScene(GraphsModel* pModel, RAPIDJSON_WRITER& writer)
{
	//QJsonArray item = { "clearAllState" };
    //ret.push_back(item);

	QStringList graphs;
	for (int i = 0; i < pModel->rowCount(); i++)
		graphs.push_back(pModel->subGraph(i)->name());

    SubGraphModel* pSubModel = pModel->subGraph("main");
    serializeGraph(pSubModel, pModel, graphs, writer, "");
}


static void appendSerializedCharArray(QString &res, const char *buf, size_t len) {
    for (auto p = buf; p < buf + len; p++) {
        res.append(QString::number((int)*p));
        res.append(',');
    }
}

QString translateGraphToCpp(const char *subgJson, size_t subgJsonLen, GraphsModel *model)
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
        auto subg = model->subGraph(i);
        auto key = subg->name();
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
    static const char mydata[] = {)RAW");
        appendSerializedCharArray(res, subgJson, subgJsonLen);
        res.append(R"RAW(};

    virtual const char *get_subgraph_json() override {
        return mydata;
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
            // FIXME: UiHelper::variantToString seems doesn't support vec3f
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
        res.append(R"RAW(}
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
