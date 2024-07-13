#include <zeno/core/ReferManager.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>
#include <zeno/core/CoreParam.h>
#include <zeno/core/INode.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/utils/helper.h>
#include <regex>

namespace zeno {
    ReferManager::ReferManager() : m_bModify(false)
    {
    }

    ReferManager::~ReferManager()
    {
    }

    void ReferManager::init(const std::shared_ptr<Graph>& pGraph)
    {
        for (auto& [key, spAny] : pGraph->getNodes())
        {
            auto spNode = AnyToINodePtr(spAny);
            if (SubnetNode* subnetNode = dynamic_cast<SubnetNode*>(spNode))
            {
                init(subnetNode->subgraph);
            }
            auto objPath = spNode->get_uuid_path();
            auto uuid_path = zeno::objPathToStr(objPath);
            auto namePath = spNode->get_path();
            namePath.pop_back();
            auto currPath = zeno::objPathToStr(namePath);
            PrimitiveParams params = spNode->get_input_primitive_params();
            for (ParamPrimitive& param : params)
            {
                auto paths = referPaths(currPath, param.defl);
                if (!paths.empty())
                {
                    auto uuid_param = uuid_path + "/" + param.name;
                    addReferInfo(paths, uuid_param);
                }
            }
        }
    }

    void zeno::ReferManager::checkReference(const ObjPath& objPath, const std::string& param)
    {
        auto spNode = AnyToINodePtr(getSession().mainGraph->getNodeByUuidPath(objPath));
        if (!spNode)
            return;

        bool bExist = false;
        ParamPrimitive paramprim = spNode->get_input_prim_param(param, &bExist);
        if (!bExist)
            return;

        auto namePath = spNode->get_path();
        namePath.pop_back();
        auto currPath = zeno::objPathToStr(namePath);
        auto uuid_path = zeno::objPathToStr(objPath);
        auto paths = referPaths(currPath, paramprim.defl);
        updateReferedInfo(uuid_path, param, paths);
        //被引用的参数数据更新时，引用该参数的节点需要标脏
        updateDirty(uuid_path, param);
    }

    void zeno::ReferManager::removeReference(const std::string& path, const std::string& uuid_path, const std::string& param)
    {
        //若删除的节点/参数被引用了
        std::set<std::string> updateParams;
        bool bRemoveNode = param.empty();
        for (auto iter = m_referInfos.begin(); iter != m_referInfos.end();)
        {
            if ((bRemoveNode && !starts_with(iter->first, uuid_path)) || (!bRemoveNode && iter->first != uuid_path))
            {
                iter++;
                continue;
            }
            if (bRemoveNode)
            {
                for (auto &[param_key, paths] : iter->second)
                {
                    updateParams.insert(paths.begin(), paths.end());
                }
                iter = m_referInfos.erase(iter);
                continue;
            }
            else
            {
                auto param_it = iter->second.find(param);
                if (param_it != iter->second.end())
                {
                    updateParams.insert(param_it->second.begin(), param_it->second.end());
                    m_referInfos[iter->first].erase(param_it);
                    if (m_referInfos[iter->first].empty())
                    {
                        iter = m_referInfos.erase(iter);
                        continue;
                    }
                }
            }
            iter++;
        }
        for (auto &uuid_param : updateParams)
        {
            int idx = uuid_param.find_last_of("/");
            auto path_str = uuid_param.substr(0, idx);
            auto param = uuid_param.substr(idx + 1, uuid_param.size() - idx);
            auto objPath = zeno::strToObjPath(path_str);
            auto spNode = AnyToINodePtr(getSession().mainGraph->getNodeByUuidPath(objPath));
            if (!spNode)
                continue;

            bool bExist = false;
            ParamPrimitive paramprim = spNode->get_input_prim_param(param, &bExist);
            if (!bExist)
                continue;

            std::string currPath = zeno::objPathToStr(spNode->get_path());
            currPath = currPath.substr(0, currPath.find_last_of("/"));
            auto val = paramprim.defl;
            if (updateParamValue(path, "0", currPath, val))
            {
                spNode->update_param(param, val);
                zeno::log_warn("the value of {} has been reseted", uuid_param);
            }
        }
        //若删除的节点/参数引用了其他参数
        for (auto& iter = m_referInfos.begin(); iter != m_referInfos.end();)
        {
            for (auto param_it = iter->second.begin(); param_it != iter->second.end();)
            {
                for (auto it = param_it->second.begin(); it != param_it->second.end();)
                {
                    std::string left_str = *it;
                    std::string right_str;
                    if (bRemoveNode)
                    {
                        left_str = left_str.substr(0, left_str.find_last_of("/"));
                        right_str = uuid_path;
                    }
                    else 
                    {
                        right_str = uuid_path + "/" + param;
                    }
                    if (starts_with(left_str, right_str))
                    {
                        it = m_referInfos[iter->first][param_it->first].erase(it);
                    }
                    else
                    {
                        it++;
                    }
                }
                if (m_referInfos[iter->first][param_it->first].empty())
                {
                    param_it = m_referInfos[iter->first].erase(param_it);
                }
                else
                {
                    param_it++;
                }
            }
            if (m_referInfos[iter->first].empty())
            {
                iter = m_referInfos.erase(iter);
            }
            else
            {
                iter++;
            }
        }
    }

    void ReferManager::addReferInfo(const std::set<std::pair<std::string, std::string>>& referedParams, const std::string& referPath)
    {
        for (const auto&param : referedParams)
        {
            if (m_referInfos.find(param.first) == m_referInfos.end())
                m_referInfos[param.first] = std::map<std::string, std::set<std::string> >();
            if (m_referInfos[param.first].find(param.second) == m_referInfos[param.first].end())
                m_referInfos[param.first][param.second] = std::set<std::string>();
            m_referInfos[param.first][param.second].emplace(referPath);
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
                for (auto &[param_key, paths] : it->second)
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
            auto spNode = AnyToINodePtr(getSession().mainGraph->getNodeByUuidPath(objPath));
            if (!spNode)
            {
                continue;
            }
            bool bExist = false;
            ParamPrimitive primparam = spNode->get_input_prim_param(param, &bExist);
            if (!bExist) {
                continue;
            }
            auto val = primparam.defl;
            std::string currentPath = zeno::objPathToStr(spNode->get_path());
            currentPath = currentPath.substr(0, currentPath.find_last_of("/"));
            bool bUpate = updateParamValue(oldPath, newPath, currentPath, val);
            if (bUpate)
            {
                //update param value
                spNode->update_param(primparam.name, val);
            }
        }
    }

    void zeno::ReferManager::updateReferedInfo(const std::string& uuid_path, const std::string& param, const std::set<std::pair<std::string, std::string>>& referedParams)
    {
        if (m_bModify)
            return;
        m_bModify = true;
        //get all refered params
        auto uuid_param = uuid_path + "/" + param;
        std::set<std::pair<std::string, std::string>> referedParams_old = getAllReferedParams(uuid_param);
        //remove info
        for (auto& it : referedParams_old)
        {
            if (referedParams.find(it) == referedParams.end())
            {
                m_referInfos[it.first][it.second].erase(uuid_param);
                if (m_referInfos[it.first][it.second].empty())
                    m_referInfos[it.first].erase(it.second);
                if (m_referInfos[it.first].empty())
                    m_referInfos.erase(it.first);
            }
        }
        //add info
        std::set<std::pair<std::string, std::string>> referedParams_add;
        for (auto &it : referedParams)
        {
            if (referedParams_old.find(it) == referedParams_old.end())
            {
                referedParams_add.emplace(it);
            }
        }
        if (!referedParams_add.empty())
            addReferInfo(referedParams_add, uuid_param);
        m_bModify = false;
    }

    bool ReferManager::isReferSelf(const std::string& uuid_path, const std::string& param) const
    {
        auto uuid_param = uuid_path + "/" + param;
        auto referSelf = [uuid_param, this](const auto& referSelf, const std::string& key)->bool {
            auto referedParams = getAllReferedParams(key);
            for (auto& it : referedParams)
            {
                auto path = it.first + "/" + it.second;
                if (path == uuid_param || referSelf(referSelf, path))
                    return true;
            }
            return false;
        };
        return referSelf(referSelf, uuid_param);
    }


    void ReferManager::updateDirty(const std::string& uuid_path, const std::string& param)
    {
        auto iter = m_referInfos.find(uuid_path);
        if (iter != m_referInfos.end())
        {
            auto param_it = iter->second.find(param);
            if (param_it != iter->second.end())
            {
                if (isReferSelf(uuid_path, param))
                {
                    zeno::log_error("{} refer loop", param);
                    return;
                }
                for (auto& path : param_it->second)
                {
                    int idx = path.find_last_of("/");
                    auto nodePath = path.substr(0, idx);
                    auto param = path.substr(idx + 1, path.size() - idx);
                    auto objPath = zeno::strToObjPath(nodePath);
                    auto spNode = AnyToINodePtr(getSession().mainGraph->getNodeByUuidPath(objPath));
                    if (!spNode)
                        continue;
                    if (!spNode->is_dirty())
                        spNode->mark_dirty(true);
                    //该节点被其他参数引用的情况下，也要标脏
                    updateDirty(nodePath, param);
                }
            }
        }
    }

    std::set <std::pair<std::string, std::string>> ReferManager::referPaths(const std::string& currPath, const zvariant& val) const
    {
        std::set <std::pair<std::string, std::string>> res;
        std::set<std::string> paths = zeno::getReferPaths(val);
        for (auto& val : paths)
        {
            std::string absolutePath = zeno::absolutePath(currPath, val);
            int idx = absolutePath.find_last_of("/");
            std::string path = absolutePath.substr(0, idx);
            std::string param = absolutePath.substr(idx + 1, val.size() - idx);
            if (auto spNode = AnyToINodePtr(getSession().mainGraph->getNodeByPath(path)))
            {
                path = zeno::objPathToStr(spNode->get_uuid_path());
                res.emplace(std::make_pair(path, param));
            }
        }
        return res;
    }


    bool ReferManager::updateParamValue(const std::string& oldVal, const std::string& newVal, const std::string& currentPath, zvariant& arg)
    {
        bool bUpdate = false;
        std::visit([oldVal, newVal, currentPath, &bUpdate](auto&& arg) {
            auto updateValue = [oldVal, newVal, currentPath, &bUpdate](auto&& arg) {
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
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>) {
                updateValue(arg);
            }
            else if constexpr (std::is_same_v<T, zeno::vec2s> || std::is_same_v<T, zeno::vec3s> || std::is_same_v<T, zeno::vec4s>)
            {
                for (int i = 0; i < arg.size(); i++)
                {
                    updateValue(arg[i]);
                }
            }
        }, arg);
        return bUpdate;
    }
}