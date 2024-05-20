#ifndef __CORE_REFERMANAGER_H__
#define __CORE_REFERMANAGER_H__

#include <map>
#include <set>
#include <string>
#include <memory>
#include "common.h"
#include <zeno/utils/api.h>

namespace zeno {
    struct INode;
    struct IParam;
    struct Graph;

    struct ReferManager
    {
    public:
        ReferManager();
        ~ReferManager();
        ZENO_API void init(const std::shared_ptr<Graph>& pGraph);
        void addReferInfo(std::shared_ptr <IParam> spParam);
        //当删除了引用了其他参数的节点后，需删除对应信息
        void removeReferParam(const std::string& uuid_param);
        //当删除了被引用的节点后，需删除对应信息
        void removeBeReferedParam(const std::string& uuid_param, const std::string& path);
        //当被引用的节点名称修改后，需要更新数据
        void updateReferParam(const std::string& oldPath, const std::string& newPath);
        //当引用的参数修改后，需要更新m_referedUuidParams
        void updateBeReferedParam(const std::string& uuid_param);
        //被引用的参数更新时需要对引用的节点标脏
        void updateDirty(const std::string& uuid_param);

        bool isRefered(const std::string& key) const;//是否引用其它参数
        bool isBeRefered(const std::string& key) const;//参数是否被引用
        bool isReferSelf(const std::string& key) const;//是否循环引用

    private:
        std::set<std::string> referPaths(const std::string& currPath, const zvariant& val) const;
        bool updateParamValue(const std::string& oldVal, const std::string& newVal, const std::string& currentPath, zvariant& arg);

        std::map <std::string, std::shared_ptr<IParam>> m_referParams;//<引用参数uuid/param, 参数ptr>
        std::map <std::string, std::set<std::string>> m_referedUuidParams;//<被引用参数uuid/param, 引用参数uuid/param 集合>
        bool m_bModify;

    };
}
#endif