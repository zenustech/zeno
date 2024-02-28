#include <zeno/io/zenwriter.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/utils/helper.h>
#include <zeno/io/iohelper.h>

using namespace zeno::iotags;


namespace zenoio
{
    ZENO_API ZenWriter::ZenWriter()
    {
    }

    ZENO_API std::string ZenWriter::dumpToClipboard(const zeno::GraphData& graph)
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

    ZENO_API std::string ZenWriter::dumpProgramStr(zeno::GraphData maingraph, AppSettings settings)
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
}