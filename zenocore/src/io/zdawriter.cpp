#include <zeno/io/zdawriter.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/utils/helper.h>
#include <zeno/io/iohelper.h>
#include <format>

using namespace zeno::iotags;


namespace zenoio
{
    ZENO_API ZdaWriter::ZdaWriter()
    {
    }

    ZENO_API std::string ZdaWriter::dumpAsset(zeno::ZenoAsset asset)
    {
        std::string strJson;

        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);

        {
            JsonObjScope batch(writer);

            writer.Key("name");
            writer.String(asset.info.name.c_str());

            writer.Key("version");
            std::string ver = zeno::format("{}.{}", asset.info.majorVer, asset.info.minorVer);
            writer.String(ver.c_str());

            if (asset.optGraph.has_value())
            {
                writer.Key("graph");
                dumpGraph(asset.optGraph.value(), writer);
            }

            writer.Key("Parameters");
            {
                JsonObjScope batch(writer);
                writer.Key("inputs");
                {
                    JsonObjScope _batch(writer);
                    for (auto param : asset.inputs)
                    {
                        writer.Key(param.name.c_str());
                        dumpSocket(param, writer);
                    }
                }
                writer.Key("outputs");
                {
                    JsonObjScope _batch(writer);
                    for (auto param : asset.outputs)
                    {
                        writer.Key(param.name.c_str());
                        dumpSocket(param, writer);
                    }
                }
            }
        }

        strJson = s.GetString();
        return strJson;
    }
}