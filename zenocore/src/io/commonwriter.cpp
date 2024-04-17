#include <zeno/io/commonwriter.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/utils/helper.h>
#include <zeno/io/iohelper.h>

using namespace zeno::iotags;


namespace zenoio
{
    CommonWriter::CommonWriter()
    {
    }

    void CommonWriter::dumpGraph(zeno::GraphData graph, RAPIDJSON_WRITER& writer)
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

    void CommonWriter::dumpNode(const zeno::NodeData& node, RAPIDJSON_WRITER& writer)
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
            if (node.bView) {
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

        if (node.asset.has_value()) {
            zeno::AssetInfo asset = node.asset.value();
            writer.Key("asset");
            JsonObjScope scope(writer);
            writer.Key("name");
            writer.String(asset.name.c_str());
            writer.Key("path");
            writer.String(asset.path.c_str());
            writer.Key("version");
            std::string version = std::to_string(asset.majorVer) + "." + std::to_string(asset.minorVer);
            writer.String(version.c_str());
        }
    }

    void CommonWriter::dumpSocket(zeno::ParamInfo param, RAPIDJSON_WRITER& writer)
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

                    writer.Key("property");
                    if (link.lnkfunc == zeno::Link_Copy) {
                        writer.String("copy");
                    }
                    else {
                        writer.String("ref");
                    }
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

            writer.Key("socket-type");
            if (param.socketType == zeno::ParamSocket) {
                writer.String("parameter");
            }
            else if (param.socketType == zeno::PrimarySocket) {
                writer.String("primary");
            }
            else {
                writer.String("none");
            }
        }

        if (!param.tooltip.empty())
        {
            writer.Key("tooltip");
            writer.String(param.tooltip.c_str());
        }
        writer.EndObject();
    }

    void CommonWriter::dumpTimeline(zeno::TimelineInfo info, RAPIDJSON_WRITER& writer)
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