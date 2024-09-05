#include <zeno/core/ReferManager.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>
#include <zeno/core/CoreParam.h>
#include <zeno/core/INode.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/utils/helper.h>
#include <zeno/formula/formula.h>
#include <zeno/core/FunctionManager.h>
#include <zeno/extra/GraphException.h>
#include <regex>


namespace zeno {

    ReferManager::ReferManager() : m_bModify(false)
    {
    }

    ReferManager::~ReferManager()
    {
    }

    void ReferManager::removeReference(const std::string& uuid_path, const std::string& param)
    {
        //若删除的节点/参数被引用了
        std::set<std::string> updateParams;
        bool bRemoveNode = param.empty();

        auto iterSourceNode = m_referInfos.find(uuid_path);
        if (iterSourceNode == m_referInfos.end())
            return;

        if (bRemoveNode) {
            m_referInfos.erase(iterSourceNode);
            return;
        }
        else
        {
            auto iterParam = iterSourceNode->second.find(param);
            if (iterParam != iterSourceNode->second.end()) {
                iterSourceNode->second.erase(iterParam);
            }
        }
    }

    void ReferManager::addReferInfo(const std::set<std::pair<std::string, std::string>>& referSources, const std::string& referPath)
    {
        //referPath的格式是： uuid-path-of-node/param.
        for (const auto& param : referSources)
        {
            const std::string& source_node_uuid = param.first;
            const std::string& source_param = param.second;

            if (m_referInfos.find(source_node_uuid) == m_referInfos.end())
                m_referInfos[source_node_uuid] = std::map<std::string, std::set<std::string> >();

            if (m_referInfos[source_node_uuid].find(source_param) == m_referInfos[source_node_uuid].end())
                m_referInfos[source_node_uuid][source_param] = std::set<std::string>();

            m_referInfos[source_node_uuid][source_param].emplace(referPath);
        }
    }

    std::set<std::pair<std::string, std::string>> zeno::ReferManager::getAllReferedParams(const std::string& uuid_param) const
    {
        std::set<std::pair<std::string, std::string>> referedParams;
        for (auto& [key, val] : m_referInfos)
        {
            for (auto& [param_key, paths] : val)
            {
                for (auto& path : paths)
                {
                    if (uuid_param == path)
                        referedParams.emplace(std::make_pair(key, param_key));
                }
            }
        }
        return referedParams;
    }

    void ReferManager::updateReferParam(const std::string& oldPath, const std::string& newPath, const std::string& uuid_path, const std::string& param)
    {
        bool bUpdateNode = param.empty();
        std::set<std::string> referParams;
        for (auto it = m_referInfos.begin(); it != m_referInfos.end(); it++)
        {
            if ((bUpdateNode && !starts_with(it->first, uuid_path)) || (!bUpdateNode && it->first != uuid_path))
                continue;
            if (bUpdateNode)
            {
                for (auto& [param_key, paths] : it->second)
                {
                    referParams.insert(paths.begin(), paths.end());
                }
            }
            else
            {
                const auto& param_it = it->second.find(param);
                if (param_it != it->second.end())
                {
                    referParams.insert(param_it->second.begin(), param_it->second.end());
                }
            }
        }
        for (const auto& uuid_param : referParams)
        {
            auto idx = uuid_param.find_last_of("/");
            auto uuid_path = uuid_param.substr(0, idx);
            auto param = uuid_param.substr(idx + 1, uuid_param.size() - idx);
            auto objPath = zeno::strToObjPath(uuid_path);
            auto spNode = getSession().mainGraph->getNodeByUuidPath(objPath);
            assert(spNode);
            bool bExist = false;
            ParamPrimitive primparam = spNode->get_input_prim_param(param, &bExist);
            assert(bExist);

            std::string currentPath = zeno::objPathToStr(spNode->get_path());
            std::string currGraph = currentPath.substr(0, currentPath.find_last_of("/"));
            bool bUpate = updateParamValue(oldPath, newPath, currGraph, primparam.defl);
            if (bUpate)
            {
                //update param value
                spNode->update_param(primparam.name, primparam.defl);
            }
        }
    }

    bool ReferManager::unregisterRelations(
        const std::string& refsource_node,
        const std::string& refsource_param,
        const std::string& refnode,
        const std::string& refparam
    )
    {
        auto iterSourceNode = m_referInfos.find(refsource_node);
        if (iterSourceNode == m_referInfos.end())
            return false;

        auto iterSourceParam = iterSourceNode->second.find(refsource_param);
        if (iterSourceParam == iterSourceNode->second.end())
            return false;

        std::string path = refnode + '/' + refparam;
        if (iterSourceParam->second.find(path) == iterSourceParam->second.end())
            return false;

        iterSourceParam->second.erase(path);
        return true;
    }

    bool zeno::ReferManager::registerRelations(
        const std::string& refnode_uuidpath,
        const std::string& referparam,
        const std::set<std::pair<std::string, std::string>>& referSources)
    {
        if (m_bModify)
            return true;

        m_bModify = true;
        //get all refered params
        auto uuid_param = refnode_uuidpath + "/" + referparam;
        std::set<std::pair<std::string, std::string>> referedParams_old = getAllReferedParams(uuid_param);
        //remove info
        for (auto& it : referedParams_old)
        {
            if (referSources.find(it) == referSources.end())
            {
                m_referInfos[it.first][it.second].erase(uuid_param);
                if (m_referInfos[it.first][it.second].empty())
                    m_referInfos[it.first].erase(it.second);
                if (m_referInfos[it.first].empty())
                    m_referInfos.erase(it.first);
            }
        }
        //add info
        std::set<std::pair<std::string, std::string>> referSources_add;
        for (auto& it : referSources)
        {
            if (referedParams_old.find(it) == referedParams_old.end())
            {
                referSources_add.emplace(it);
            }
        }
        if (!referSources_add.empty()) {
            addReferInfo(referSources_add, uuid_param);
        }
        m_bModify = false;
        return true;
    }

    void ReferManager::updateDirty(const std::string& uuid_path_sourcenode, const std::string& param)
    {
        auto iter = m_referInfos.find(uuid_path_sourcenode);
        if (iter != m_referInfos.end())
        {
            auto param_it = iter->second.find(param);
            if (param_it != iter->second.end())
            {
                for (auto& path : param_it->second)
                {
                    size_t idx = path.find_last_of("/");
                    auto nodePath = path.substr(0, idx);
                    auto param = path.substr(idx + 1, path.size() - idx);
                    auto objPath = zeno::strToObjPath(nodePath);
                    auto spNode = getSession().mainGraph->getNodeByUuidPath(objPath);
                    assert(spNode);
                    if (!spNode->is_dirty())
                    {
                        spNode->mark_dirty(true);
                        //该节点被其他参数引用的情况下，也要标脏
                        updateDirty(nodePath, param);
                    }
                }
            }
        }
    }

    bool ReferManager::updateParamValue(
            const std::string& oldVal,
            const std::string& newVal,
            const std::string& currentPath,
            zeno::reflect::Any& adjustParamVal) const
    {
        bool bUpdate = false;

        auto fUpdateParamDefl = [oldVal, newVal, currentPath, &bUpdate](std::string& arg) {
            auto matchs = zeno::getReferPath(arg);
            for (const auto& str : matchs)
            {
                std::string absolutePath = zeno::absolutePath(currentPath, str);
                if (absolutePath.find(oldVal) != std::string::npos)
                {
                    std::regex num_rgx("[0-9]+");
                    //如果是数字，需要将整个refer替换
                    if (std::regex_match(newVal, num_rgx))
                    {
                        arg = newVal;
                        bUpdate = true;
                        break;
                    }
                    else
                    {
                        std::regex pattern(oldVal);
                        std::string format = regex_replace(absolutePath, pattern, newVal);
                        //relative path
                        if (absolutePath != str)
                        {
                            format = zeno::relativePath(currentPath, format);
                        }
                        std::regex rgx(str);
                        arg = regex_replace(arg, rgx, format);
                    }
                    bUpdate = true;
                }
            }
        };

        assert(adjustParamVal.has_value());
        ParamType type = adjustParamVal.type().hash_code();
        if (type == zeno::types::gParamType_PrimVariant) {
            PrimVar& var = zeno::reflect::any_cast<PrimVar>(adjustParamVal);
            std::visit([&](auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    fUpdateParamDefl(arg);
                }
                else {
                    assert(false);
                    zeno::log_warn("error param type");
                }
            }, var);
            if (bUpdate) {
                adjustParamVal = zeno::reflect::move(var);
            }
        }
        else if (type == zeno::types::gParamType_VecEdit) {
            vecvar var = zeno::reflect::any_cast<vecvar>(adjustParamVal);
            for (PrimVar& elem : var)
            {
                std::visit([&](auto& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::string>) {
                        fUpdateParamDefl(arg);
                    }
                }, elem);
            }
            if (bUpdate) {
                adjustParamVal = zeno::reflect::move(var);
            }
        }
        else {
            assert(false);
            zeno::log_error("unknown param type of refer param");
        }
        return bUpdate;
    }
}