#include "zsgwriter.h"
#include <zenoui/model/modelrole.h>
#include <zeno/utils/logger.h>


ZsgWriter::ZsgWriter()
{
}

ZsgWriter& ZsgWriter::getInstance()
{
	static ZsgWriter writer;
	return writer;
}

QString ZsgWriter::dumpProgramStr(IGraphsModel* pModel)
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
				const QString& subgName = pModel->name(subgIdx);
				writer.Key(subgName.toLatin1());
				_dumpSubGraph(pModel, subgIdx, writer);
			}
		}

		writer.Key("views");
		{
			writer.StartObject();
			writer.EndObject();
		}

		NODE_DESCS descs = pModel->descriptors();
		writer.Key("descs");
		_dumpDescriptors(descs, writer);

		writer.Key("version");
		writer.String("v2");
	}

	strJson = QString::fromLatin1(s.GetString());
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
		const NODES_DATA& nodes = pModel->nodes(subgIdx);
		for (const NODE_DATA& node : nodes)
		{
			const QString& id = node[ROLE_OBJID].toString();
			writer.Key(id.toLatin1());
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

void ZsgWriter::dumpNode(const NODE_DATA& data, RAPIDJSON_WRITER& writer)
{
	JsonObjBatch batch(writer);

	writer.Key("name");
	const QString& name = data[ROLE_OBJNAME].toString();
	writer.String(name.toLatin1());

	writer.Key("inputs");
	{
		JsonObjBatch _batch(writer);

		const INPUT_SOCKETS& inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
		for (const INPUT_SOCKET& inSock : inputs)
		{
			writer.Key(inSock.info.name.toLatin1());

			QVariant deflVal = inSock.info.defaultValue;
			if (!inSock.linkIndice.isEmpty())
			{
				for (QPersistentModelIndex linkIdx : inSock.linkIndice)
				{
					QString outNode = linkIdx.data(ROLE_OUTNODE).toString();
					QString outSock = linkIdx.data(ROLE_OUTSOCK).toString();
					AddVariantListWithNull({ outNode, outSock, deflVal }, writer);
				}
			}
			else
			{
				AddVariantListWithNull({ QVariant(), QVariant(), deflVal}, writer);
			}
		}
	}
	writer.Key("params");
	{
		JsonObjBatch _batch(writer);

		const PARAMS_INFO& params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();
		for (const PARAM_INFO& info : params)
		{
			writer.Key(info.name.toLatin1());

			QVariant val = info.value;
			if (val.type() == QVariant::String) {
				writer.String(val.toString().toLatin1());
			}
			else if (val.type() == QVariant::Double) {
				writer.Double(val.toDouble());
			}
			else if (val.type() == QVariant::Int) {
				writer.Int(val.toInt());
			}
			else if (val.type() == QVariant::Bool) {
				writer.Bool(val.toBool());
			}
			else if (val.type() != QVariant::Invalid) {
				zeno::log_warn("bad param info qvariant type {}", val.typeName() ? val.typeName() : "(null)");
				writer.String("");	//to think...
			}
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
	
	QStringList socketKeys = data[ROLE_SOCKET_KEYS].value<QStringList>();
	if (!socketKeys.isEmpty())
	{
		writer.Key("socket_keys");
		AddStringList(socketKeys, writer);
	}
}

void ZsgWriter::_dumpDescriptors(const NODE_DESCS& descs, RAPIDJSON_WRITER& writer)
{
	JsonObjBatch batch(writer);

	for (auto name : descs.keys())
	{
		const NODE_DESC& desc = descs[name];

		writer.Key(name.toLatin1());
		JsonObjBatch _batch(writer);

		{
			writer.Key("inputs");
			JsonArrayBatch _batchArr(writer);

			for (INPUT_SOCKET inSock : desc.inputs)
			{
				AddVariantToStringList({ inSock.info.type ,inSock.info.name, inSock.info.defaultValue }, writer);
			}
		}
		{
			writer.Key("params");
			JsonArrayBatch _batchArr(writer);

			for (PARAM_INFO param : desc.params)
			{
				AddVariantToStringList({ param.typeDesc , param.name, param.defaultValue }, writer);
			}
		}

		{
			writer.Key("outputs");
			JsonArrayBatch _batchArr(writer);

			for (OUTPUT_SOCKET outSock : desc.outputs)
			{
				AddVariantToStringList({ outSock.info.type , outSock.info.name, outSock.info.defaultValue }, writer);
			}
		}

		writer.Key("categories");
		AddStringList(desc.categories, writer);

		if (desc.is_subgraph)
		{
			writer.Key("is_subgraph");
			writer.Bool(true);
		}
	}
}
