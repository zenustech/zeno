#include "ZenWriter.h"
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/utils/helper.h>
#include <zenoio/include/iohelper.h>

using namespace zeno::iotags;


namespace zenoio
{
    ZenWriter::ZenWriter()
    {
    }

    std::string ZenWriter::dumpToClipboard(const zeno::GraphData& graph)
    {
        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);
        {
            JsonObjScope batch(writer);
            writer.Key("nodes");
            {
                JsonObjScope _batch(writer);
                for (const auto& [name, node_] : graph.nodes)
                {
                    if (node_.type == zeno::NoVersionNode) {
                        continue;
                    }
                    writer.Key(name.c_str());
                    dumpNode(node_, writer);
                }
            }
        }
        std::string strJson = s.GetString();
        return strJson;
    }

    std::string ZenWriter::dumpProgramStr(zeno::GraphData maingraph, AppSettings settings)
    {
        std::string strJson;

        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);

        {
            JsonObjScope batch(writer);

            writer.Key("main");
            dumpGraph(maingraph, writer);

            writer.Key("views");
            {
                writer.StartObject();
                dumpTimeline(settings.timeline, writer);
                writer.EndObject();
            }

            writer.Key("version");
            writer.String("v3");
        }

        strJson = s.GetString();
        return strJson;
    }


    void ZenWriter::dumpGraph(zeno::GraphData graph, RAPIDJSON_WRITER& writer)
    {
        JsonObjScope batch(writer);
        {
            writer.Key("type");
            writer.Int(graph.type);
            writer.Key("nodes");
            JsonObjScope _batch(writer);

            for (auto& [name, node] : graph.nodes)
            {
                if (node.type == zeno::NoVersionNode)
                    continue;
                writer.Key(name.c_str());
                dumpNode(node, writer);
            }
        }
    }

    void ZenWriter::dumpNode(const zeno::NodeData& node, RAPIDJSON_WRITER& writer)
    {
        JsonObjScope batch(writer);

        writer.Key("name");
        writer.String(node.name.c_str());

        writer.Key("class");
        writer.String(node.cls.c_str());

        writer.Key("inputs");
        {
            JsonObjScope _batch(writer);
            for (const auto& param : node.inputs)
            {
                writer.Key(param.name.c_str());
                dumpSocket(param, writer);
            }
        }
        writer.Key("outputs");
        {
            JsonObjScope _scope(writer);
            for (const auto& param : node.outputs)
            {
                writer.Key(param.name.c_str());
                dumpSocket(param, writer);
            }
        }
        writer.Key("uipos");
        {
            writer.StartArray();
            writer.Double(node.uipos.first);    //x
            writer.Double(node.uipos.second);   //y
            writer.EndArray();
        }

        writer.Key("status");
        {
            std::vector<std::string> options;
            if (node.status & zeno::Mute) {
                options.push_back("MUTE");
            }
            if (node.status & zeno::View) {
                options.push_back("View");
            }
            writer.StartArray();
            for (auto item : options)
            {
                writer.String(item.c_str(), item.length());
            }
            writer.EndArray();
        }

        writer.Key("collasped");
        writer.Bool(node.bCollasped);

        if (node.cls == "Blackboard") {
            // do not compatible with zeno1

            for (auto param : node.inputs)
            {
                //TODO
            }
        }
        else if (node.cls == "Group") {
            // TODO
        }
        //TODO: custom ui for panel

        if (node.subgraph.has_value() && node.type == zeno::Node_SubgraphNode)
        {
            writer.Key("subnet");
            dumpGraph(node.subgraph.value(), writer);
        }

        //TODO: assets
    }

    void ZenWriter::dumpSocket(zeno::ParamInfo param, RAPIDJSON_WRITER& writer)
    {
        //new io format for socket.
        writer.StartObject();

        zeno::SocketProperty prop;
        param.prop;
        //property
        if (param.prop != zeno::Socket_Normal)
        {
            writer.Key("property");
            if (param.prop & zeno::Socket_Editable) {
                writer.String("editable");
            }
            else if (param.prop & zeno::Socket_Legacy) {
                writer.String("legacy");
            }
            else {
                writer.String("normal");
            }
        }

        if (param.bInput)
        {
            writer.Key("links");
            if (param.links.empty())
            {
                writer.Null();
            }
            else
            {
                writer.StartArray();
                for (auto link : param.links) {

                    JsonObjScope scope(writer);
                    writer.Key("out-node");
                    writer.String(link.outNode.c_str());
                    writer.Key("out-socket");
                    writer.String(link.outParam.c_str());
                    writer.Key("out-key");
                    writer.String(link.outKey.c_str());

                    writer.Key("in-key");
                    writer.String(link.inKey.c_str());
                }
                writer.EndArray();
            }
        }

        writer.Key("type");
        writer.String(zeno::paramTypeToString(param.type).c_str());

        if (param.bInput)
        {
            writer.Key("default-value");
            writeZVariant(param.defl, param.type, writer);

            writer.Key("control");
            dumpControl(param.type, param.control, param.ctrlProps, writer);
        }

        if (!param.tooltip.empty())
        {
            writer.Key("tooltip");
            writer.String(param.tooltip.c_str());
        }
        writer.EndObject();
    }

    void ZenWriter::dumpTimeline(zeno::TimelineInfo info, RAPIDJSON_WRITER& writer)
    {
        writer.Key("timeline");
        {
            JsonObjScope _batch(writer);
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
}