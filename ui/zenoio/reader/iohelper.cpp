#include "../include/iohelper.h"
#include <zeno/zeno.h>


namespace zenoio
{
    zeno::GraphData fork(
        const std::string& currentPath,
        const zeno::AssetsData& subgraphDatas,
        const std::string& subnetName)
    {
        zeno::GraphData newGraph;
        zeno::NodesData newDatas;
        zeno::LinksData newLinks;

        std::map<std::string, zeno::NodeData> nodes;
        std::unordered_map<std::string, std::string> old2new;
        zeno::LinksData oldLinks;

        if (subgraphDatas.find(subnetName) == subgraphDatas.end())
        {
            return newGraph;
        }

        const zeno::GraphData& subgraph = subgraphDatas[subnetName];

        for (const auto& [ident, nodeData] : subgraph.nodes)
        {
            zeno::NodeData nodeDat = nodeData;
            const std::string& snodeId = nodeDat.ident;
            const std::string& name = nodeDat.cls;
            const std::string& newId = zeno::generateUUID();
            old2new.insert(snodeId, newId);

            if (subgraphDatas.find(name) != subgraphDatas.end())
            {
                const std::string& ssubnetName = name;
                nodeDat.ident = newId;

                zeno::LinksData childLinks;
                zeno::GraphData fork_subgraph;
                fork_subgraph = fork(
                    currentPath + "/" + newId,
                    subgraphDatas,
                    ssubnetName);
                fork_subgraph.links = childLinks;
                nodeDat.subgraph = fork_subgraph;

                newDatas[newId] = nodeDat;
            }
            else
            {
                nodeDat.ident = newId;
                newDatas[newId] = nodeDat;
            }
        }

        for (zeno::EdgeInfo oldLink : subgraph.links)
        {
            zeno::EdgeInfo newLink = oldLink;
            newLink.inNode = old2new[newLink.inNode];
            newLink.outNode = old2new[newLink.outNode];
            newLinks.push_back(newLink);
        }

        newGraph.nodes = nodes;
        newGraph.links = newLinks;
        return newGraph;
    }


    zeno::NodeDescs getCoreDescs()
    {
        zeno::NodeDescs descs;
        std::string strDescs = zeno::getSession().dumpDescriptors();
        //zeno::log_critical("EEEE {}", strDescs.toStdString());
        //ZENO_P(strDescs.toStdString());


        QStringList L = strDescs.split("\n");
        for (int i = 0; i < L.size(); i++)
        {
            QString line = L[i];
            if (line.startsWith("DESC@"))
            {
                line = line.trimmed();
                int idx1 = line.indexOf("@");
                int idx2 = line.indexOf("@", idx1 + 1);
                ZASSERT_EXIT(idx1 != -1 && idx2 != -1, descs);
                QString wtf = line.mid(0, idx1);
                QString z_name = line.mid(idx1 + 1, idx2 - idx1 - 1);
                QString rest = line.mid(idx2 + 1);
                ZASSERT_EXIT(rest.startsWith("{") && rest.endsWith("}"), descs);
                auto _L = rest.mid(1, rest.length() - 2).split("}{");
                QString inputs = _L[0], outputs = _L[1], params = _L[2], categories = _L[3];
                QStringList z_categories = categories.split('%', QtSkipEmptyParts);

                NODE_DESC desc;
                for (QString input : inputs.split("%", QtSkipEmptyParts)) {
                    QString type, name;
                    QVariant defl;

                    parseDescStr(input, name, type, defl);

                    INPUT_SOCKET socket;
                    socket.info.type = type;
                    socket.info.name = name;
                    CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_INPUT, name, type);
                    socket.info.control = ctrlInfo.control;
                    socket.info.ctrlProps = ctrlInfo.controlProps.toMap();
                    socket.info.defaultValue = defl;
                    desc.inputs[name] = socket;
                }
                for (QString output : outputs.split("%", QtSkipEmptyParts)) {
                    QString type, name;
                    QVariant defl;

                    parseDescStr(output, name, type, defl);

                    OUTPUT_SOCKET socket;
                    socket.info.type = type;
                    socket.info.name = name;
                    CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_OUTPUT, name, type);
                    socket.info.control = ctrlInfo.control;
                    socket.info.ctrlProps = ctrlInfo.controlProps.toMap();
                    socket.info.defaultValue = defl;
                    desc.outputs[name] = socket;
                }
                for (QString param : params.split("%", QtSkipEmptyParts)) {
                    QString type, name;
                    QVariant defl;

                    parseDescStr(param, name, type, defl);

                    PARAM_INFO paramInfo;
                    paramInfo.bEnableConnect = false;
                    paramInfo.name = name;
                    paramInfo.typeDesc = type;
                    CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_PARAM, name, type);
                    paramInfo.control = ctrlInfo.control;
                    paramInfo.controlProps = ctrlInfo.controlProps;
                    paramInfo.defaultValue = defl;
                    //thers is no "value" in descriptor, but it's convient to initialize param value.
                    paramInfo.value = paramInfo.defaultValue;
                    desc.params[name] = paramInfo;
                }
                desc.categories = z_categories;
                desc.name = z_name;

                descs.insert(z_name, desc);
            }
        }
        return descs;
    }
}