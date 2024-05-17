#include <zeno/core/ReferManager.h>
#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>
#include <zeno/core/IParam.h>
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
        for (auto& [key, spNode] : pGraph->m_nodes)
        {
            if (std::shared_ptr<SubnetNode> subnetNode = std::dynamic_pointer_cast<SubnetNode>(spNode))
            {
                init(subnetNode->subgraph);
            }
            for (auto& [key, spParam] : spNode->m_inputs)
            {
                spNode->checkReference(spParam);
            }
        }
    }

    void ReferManager::addReferInfo(std::shared_ptr <IParam> spParam)
    {
        auto spNode = spParam->m_wpNode.lock();
        if (!spNode)
            return;
        std::string key = spNode->m_uuid + "/" + spParam->name;
        if (m_referParams.find(key) != m_referParams.end())
            return;
        m_referParams[key] = spParam;
        std::string currPath = spNode->get_path_str();
        currPath = currPath.substr(0, currPath.find_last_of("/"));
        std::set<std::string> paths = referPaths(currPath, spParam->defl);
        for (const std::string& path : paths)
        {
            m_referedUuidParams[path].emplace(key);
        }
    }

    void ReferManager::removeReferParam(const std::string& uuid_param)
    {
            if (m_referParams.empty())
                return;
            std::set<std::string> removeItems;
            for (auto& [key, spParam] : m_referParams)
            {
                if (key.find(uuid_param) != std::string::npos)
                {
                    removeItems.insert(key);
                    updateBeReferedParam(key);
                }
            }
            //remove refer info
            for (const auto& item : removeItems)
            {
                m_referParams.erase(item);
            }
    }

    void ReferManager::removeBeReferedParam(const std::string& uuid_param, const std::string& path)
    {
        if (m_bModify)
            return;
        m_bModify = true;
        if (m_referedUuidParams.empty())
            return;
        std::set<std::string> removeItems;
        for (auto& [key, val] : m_referedUuidParams)
        {
            if (key.find(uuid_param) != std::string::npos)
            {
                //update param value
                for (auto& info_key : val)
                {
                    if (m_referParams.find(info_key) != m_referParams.end())
                    {
                        auto spParam = m_referParams[info_key];
                        //update param value
                        auto spNode = spParam->m_wpNode.lock();
                        std::string currPath = spNode->get_path_str();
                        currPath = currPath.substr(0, currPath.find_last_of("/"));
                        auto val = spParam->defl;
                        if (updateParamValue(path, "0", currPath, val))
                        {
                            removeItems.insert(key);
                            if (zeno::getReferPaths(val).empty())
                                m_referParams.erase(info_key);
                            spNode->update_param(spParam->name, val);
                        }
                    }
                }
            }
        }
        for (const auto& item : removeItems)
        {
            m_referedUuidParams.erase(item);
        }
        m_bModify = false;
    }

    void ReferManager::updateReferParam(const std::string& oldPath, const std::string& newPath)
    {
        for (auto& [key, spParam] : m_referParams)
        {
            if (auto spNode = spParam->m_wpNode.lock())
            {
                auto val = spParam->defl;
                std::string currentPath = spNode->get_path_str();
                currentPath = currentPath.substr(0, currentPath.find_last_of("/"));
                bool bUpate = updateParamValue(oldPath, newPath, currentPath, val);
                if (bUpate)
                {
                    //update param value
                    auto pNode = spParam->m_wpNode.lock();
                    if (pNode)
                        pNode->update_param(spParam->name, val);
                }
            }
        }
    }

    void zeno::ReferManager::updateBeReferedParam(const std::string& key)
    {
        if (m_bModify)
            return;
        m_bModify = true;
        if (m_referParams.find(key) != m_referParams.end())
        {
            if (const auto& spParam = m_referParams[key])
            {
                if (const auto& spNode = spParam->m_wpNode.lock())
                {
                    auto currPath = spNode->get_path_str();
                    currPath = currPath.substr(0, currPath.find_last_of("/"));
                    auto paths = referPaths(currPath, spParam->defl);
                    std::set<std::string> removeItems;
                    for (auto& [referKey, vals] : m_referedUuidParams)
                    {
                        for (auto& path : vals)
                        {
                            if (path == key && paths.find(referKey) == paths.end())
                                removeItems.emplace(referKey);
                        }
                    }
                    for (auto& path : paths)
                    {
                        if (m_referedUuidParams.find(path) == m_referedUuidParams.end())
                        {
                            m_referedUuidParams[path] = std::set<std::string>();
                        }
                        if (m_referedUuidParams[path].find(key) == m_referedUuidParams[path].end())
                            m_referedUuidParams[path].emplace(key);
                    }
                    for (auto& item : removeItems)
                    {
                        m_referedUuidParams[item].erase(key);
                        if (m_referedUuidParams[item].empty())
                            m_referedUuidParams.erase(item);
                    }
                }
            }
            
        }
        m_bModify = false;
    }

    bool ReferManager::isReferSelf(const std::string& key) const
    {
        auto it = m_referParams.find(key);
        while (it != m_referParams.end())
        {
            if (const auto& spParam = it->second)
            {
                if (const auto& spNode = spParam->m_wpNode.lock())
                {
                    std::string currPath = spNode->get_path_str();
                    currPath = currPath.substr(0, currPath.find_last_of("/"));
                    auto paths = referPaths(currPath, spParam->defl);
                    for (const auto& path : paths)
                    {
                        if (path == key)
                        {
                            return true;
                        }
                        else
                        {
                            it = m_referParams.find(path);
                        }
                    }
                }
            }
        }
        return false;
    }


    bool ReferManager::isRefered(const std::string& key) const
    {
        return m_referParams.find(key) != m_referParams.end();
    }

    bool zeno::ReferManager::isBeRefered(const std::string& key) const
    {
        return m_referedUuidParams.find(key) != m_referedUuidParams.end();
    }

    void ReferManager::updateDirty(const std::string& key)
    {
        auto it = m_referedUuidParams.find(key);
        if (it != m_referedUuidParams.end())
        {
            for (auto& path : it->second)
            {
                auto iter_info = m_referParams.find(path);
                if (iter_info != m_referParams.end())
                {
                    const auto& spParam = iter_info->second;
                    if (auto spNode = spParam->m_wpNode.lock())
                    {
                        spNode->mark_dirty(true);
                        //该节点被其他参数引用的情况下，也要标脏
                        updateDirty(path);
                    }
                }
            }
        }
    }

    std::set<std::string> ReferManager::referPaths(const std::string& currPath, const zvariant& val) const
    {
        std::set<std::string> res;
        std::set<std::string> paths = zeno::getReferPaths(val);
        for (auto& val : paths)
        {
            std::string absolutePath = zeno::absolutePath(currPath, val);
            int idx = absolutePath.find_last_of("/");
            std::string path = absolutePath.substr(0, idx);
            std::string param = absolutePath.substr(idx + 1, val.size() - idx);
            if (auto spNode = getSession().mainGraph->getNodeByPath(path))
            {
                path = spNode->m_uuid + "/" + param;
                res.emplace(path);
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
                            std::string s = "ref(\"" + str + "\")";
                            int idx = arg.find(s);
                            arg.replace(idx, s.size(), newVal);
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