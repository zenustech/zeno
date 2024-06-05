#include <zeno/io/zdawriter.h>
#include <zeno/utils/logger.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/utils/helper.h>
#include <zeno/io/iohelper.h>
#include <zeno/io/iotags.h>
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
            JsonObjScope scope(writer);

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

            /*writer.Key("Parameters");
            {
                JsonObjScope batch(writer);
                
                writer.Key(iotags::params::node_inputs_objs);
                {
                    JsonObjScope _scope(writer);
                    for (auto param : asset.object_inputs)
                    {
                        writer.Key(param.name.c_str());
                        dumpObjectParam(param, writer);
                    }
                }

                writer.Key(iotags::params::node_inputs_primitive);
                {
                    JsonObjScope _scope(writer);
                    for (auto param : asset.primitive_inputs)
                    {
                        writer.Key(param.name.c_str());
                        dumpPrimitiveParam(param, writer);
                    }
                }

                writer.Key(iotags::params::node_outputs_primitive);
                {
                    JsonObjScope _batch(writer);
                    for (auto param : asset.primitive_outputs)
                    {
                        writer.Key(param.name.c_str());
                        dumpPrimitiveParam(param, writer);
                    }
                }

                writer.Key(iotags::params::node_outputs_objs);
                {
                    JsonObjScope _scope(writer);
                    for (auto param : asset.object_outputs)
                    {
                        writer.Key(param.name.c_str());
                        dumpObjectParam(param, writer);
                    }
                }
            }*/
            dumpCustomUI(asset.m_customui, writer);
        }

        strJson = s.GetString();
        return strJson;
    }
}