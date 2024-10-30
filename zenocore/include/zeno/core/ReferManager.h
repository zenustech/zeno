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

    struct ReferManager
    {
    public:
        ReferManager();
        ~ReferManager();
        bool registerRelations(
            const std::string& refnode_uuidpath,
            const std::string& referparam,
            const std::set<std::pair<std::string, std::string>>& referSources);

        bool unregisterRelations(
            const std::string& refsource_node,
            const std::string& refsource_param,
            const std::string& refnode,
            const std::string& refparam
        );

        void removeReference(const std::string& uuid_path, const std::string& param = "");
        //�������õĽڵ������޸ĺ���Ҫ��������
        void updateReferParam(const std::string& oldPath, const std::string& newPath, const std::string& uuid_path, const std::string& param = "");
        void updateDirty(const std::string& uuid_path, const std::string& param);

    private:
        void addReferInfo(const std::set<std::pair<std::string, std::string>>& referSources, const std::string& referPath);
        //�����õĲ����޸ĺ���Ҫ��������

        //�����õĲ�������ʱ��Ҫ�����õĽڵ����
        std::set<std::pair<std::string, std::string>> getAllReferedParams(const std::string& uuid_param) const;

        bool updateParamValue(
            const std::string& oldPath,
            const std::string& newPath,
            const std::string& currentPath,
            zeno::reflect::Any& adjustParamVal) const;

        //<�����ò���uuidpath, <�����ò���, ���ò���params>>
        std::map <std::string, std::map<std::string, std::set<std::string> > > m_referInfos; 
        bool m_bModify;
    };
}
#endif