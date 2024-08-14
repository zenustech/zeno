#include <zeno/io/commonwriter.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/utils/helper.h>
#include <zeno/io/iohelper.h>
#include <zeno/io/iotags.h>


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

        writer.Key(iotags::params::node_inputs_objs);
        {
            JsonObjScope _batch(writer);
            for (const auto& param : node.customUi.inputObjs)
            {
                writer.Key(param.name.c_str());
                dumpObjectParam(param, writer);
            }
        }
        writer.Key(iotags::params::node_inputs_primitive);
        {
            JsonObjScope _batch(writer);
            zeno::PrimitiveParams params = customUiToParams(node.customUi.inputPrims);
            for (const auto& param : params)
            {
                writer.Key(param.name.c_str());
                dumpPrimitiveParam(param, writer);
            }
        }
        writer.Key(iotags::params::node_outputs_primitive);
        {
            JsonObjScope _batch(writer);
            zeno::PrimitiveParams params = node.customUi.outputPrims;
            for (const auto& param : params)
            {
                writer.Key(param.name.c_str());
                dumpPrimitiveParam(param, writer);
            }
        }
        writer.Key(iotags::params::node_outputs_objs);
        {
            JsonObjScope _batch(writer);
            for (const auto& param : node.customUi.outputObjs)
            {
                writer.Key(param.name.c_str());
                dumpObjectParam(param, writer);
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
        }
        else if (node.cls == "Group") {
            // TODO
        }
        else if (node.cls == "Subnet")
        {
            dumpCustomUI(node.customUi, writer);
        }

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

    void CommonWriter::dumpCustomUI(zeno::CustomUI customUi, RAPIDJSON_WRITER& writer)
    {
        writer.Key("subnet-customUi");
        JsonObjScope scopeui(writer);
        writer.Key("nickname");
        writer.String(customUi.nickname.c_str());
        writer.Key("iconResPath");
        writer.String(customUi.iconResPath.c_str());
        writer.Key("category");
        writer.String(customUi.category.c_str());
        writer.Key("doc");
        writer.String(customUi.doc.c_str());

        writer.Key("tabs");
        {
            JsonObjScope scopetabs(writer);
            for (const zeno::ParamTab& tab : customUi.inputPrims.tabs)
            {
                writer.Key(tab.name.c_str());
                JsonObjScope scopetab(writer);
                for (const zeno::ParamGroup& group : tab.groups)
                {
                    writer.Key(group.name.c_str());
                    JsonObjScope scopegroup(writer);
                    for (const zeno::ParamPrimitive& param : group.params)
                    {
                        writer.Key(param.name.c_str());
                        dumpPrimitiveParam(param, writer);
                    }
                }
            }
        }
    }

    void CommonWriter::dumpObjectParam(zeno::ParamObject param, RAPIDJSON_WRITER& writer)
    {
        JsonObjScope scope(writer);
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
                    JsonObjScope scope2(writer);
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

        //if (param.bInput)
        {
            writer.Key("socket-type");
            switch (param.socketType)
            {
            case zeno::Socket_Output:   writer.String(iotags::params::socket_output); break;
            case zeno::Socket_Clone:    writer.String(iotags::params::socket_clone); break;
            case zeno::Socket_Owning:   writer.String(iotags::params::socket_owning); break;
            case zeno::Socket_ReadOnly: writer.String(iotags::params::socket_readonly); break;
            case zeno::Socket_Primitve: writer.String(iotags::params::socket_primitive); break;
            case zeno::Socket_WildCard: writer.String(iotags::params::socket_wildcard); break;
            default:
                writer.String(iotags::params::socket_none);
            }
        }
        if (!param.wildCardGroup.empty())
        {
            writer.Key("wild_card_group");
            writer.String(param.wildCardGroup.c_str());
        }
        if (!param.tooltip.empty())
        {
            writer.Key("tooltip");
            writer.String(param.tooltip.c_str());
        }
    }

    void CommonWriter::dumpPrimitiveParam(zeno::ParamPrimitive param, RAPIDJSON_WRITER& writer)
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
            writeAny(param.defl, param.type, writer);

            writer.Key("control");
            dumpControl(param.type, param.control, param.ctrlProps, writer);

#if 0
            writer.Key("controlProps");
            {
                if (param.ctrlProps.has_value()) {
                    JsonObjScope scopeCtrlProps(writer);
                    writer.Key("items");
                    if (param.ctrlProps->items.has_value()) {
                        writer.StartArray();
                        for (auto& v : param.ctrlProps->items.value())
                            writer.String(v.c_str());
                        writer.EndArray();
                    }
                    else {
                        writer.Null();
                    }

                    writer.Key("ranges");
                    if (param.ctrlProps->ranges.has_value()) {
                        JsonObjScope scopeRange(writer);
                        writer.Key("min");
                        writer.Double(param.ctrlProps->ranges.value()[0]);
                        writer.Key("max");
                        writer.Double(param.ctrlProps->ranges.value()[1]);
                        writer.Key("step");
                        writer.Double(param.ctrlProps->ranges.value()[2]);
                    }
                    else {
                        writer.Null();
                    }
                }
                else {
                    writer.Null();
                }
            }
#endif

            writer.Key("socket-type");
            switch (param.socketType)
            {
            case zeno::Socket_Clone:    writer.String(iotags::params::socket_clone); break;
            case zeno::Socket_Owning:   writer.String(iotags::params::socket_owning); break;
            case zeno::Socket_ReadOnly: writer.String(iotags::params::socket_readonly); break;
            case zeno::Socket_Primitve: writer.String(iotags::params::socket_primitive); break;
            case zeno::Socket_WildCard: writer.String(iotags::params::socket_wildcard); break;
            default:
                writer.String(iotags::params::socket_none);
            }

        }
        writer.Key("visible");
        writer.Bool(param.bVisible);

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
            writer.Key(iotags::timeline::start_frame);
            writer.Int(info.beginFrame);
            writer.Key(iotags::timeline::end_frame);
            writer.Int(info.endFrame);
            writer.Key(iotags::timeline::curr_frame);
            writer.Int(info.currFrame);
            writer.Key(iotags::timeline::always);
            writer.Bool(info.bAlways);
            writer.Key(iotags::timeline::timeline_fps);
            writer.Int(info.timelinefps);
        }
    }
}