#ifndef __CORE_REFERMANAGER_H__
#define __CORE_REFERMANAGER_H__

#include <map>
#include <set>
#include <string>
#include <memory>
#include "common.h"
#include <zeno/utils/api.h>
#include <zeno/core/Graph.h>

namespace zeno {
    //struct Graph;

    struct ReferManager
    {
    public:
        ReferManager();
        ~ReferManager();
        ZENO_API void init(const std::shared_ptr<Graph>& pGraph);
        void registerRelations(
            const std::string& refnode_uuidpath,
            const std::string& referparam,
            const std::set<std::pair<std::string, std::string>>& referSources);

        void removeReference(const std::string& path, const std::string& uuid_path, const std::string& param = "");
        //当被引用的节点名称修改后，需要更新数据
        void updateReferParam(const std::string& oldPath, const std::string& newPath, const std::string& uuid_path, const std::string& param = "");
        bool isReferSelf(const std::string& uuid_path, const std::string& param) const;//是否循环引用
        void updateDirty(const std::string& uuid_path, const std::string& param);

    private:
        void addReferInfo(const std::set<std::pair<std::string, std::string>>& referSources, const std::string& referPath);
        //当引用的参数修改后，需要更新数据


        //被引用的参数更新时需要对引用的节点标脏
        std::set<std::pair<std::string, std::string>> getAllReferedParams(const std::string& uuid_param) const;

        std::set<std::pair<std::string, std::string>> referPaths(const std::string& currPath, const zeno::reflect::Any& val) const;
        bool updateParamValue(const std::string& oldVal, const std::string& newVal, const std::string& currentPath, zvariant& arg) const;

        //<被引用参数uuidpath, <被引用参数, 引用参数params>>
        std::map <std::string, std::map<std::string, std::set<std::string> > > m_referInfos; 
        bool m_bModify;
    };
}
#endif