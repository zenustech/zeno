#include <zeno/core/CalcManager.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GraphException.h>
#include <zeno/core/ObjectManager.h>


namespace zeno {

    CalcManager::CalcManager() {
    }

    ZENO_API void CalcManager::run() {
        auto& sess = getSession();

        sess.objsMan->beforeRun();
        zeno::scope_exit sp([&]() { sess.objsMan->afterRun(); });

        //对之前删除节点时记录的obj，对应的所有其他关联节点，都标脏
        auto& mainG = sess.mainGraph;
        for (auto obj_key : removing_objs) {
            auto nodes = sess.objsMan->getAttachNodes(obj_key);
            for (auto node_path : nodes) {
                auto spNode = mainG->getNode(node_path);
                if (spNode)
                    spNode->mark_dirty(true);
            }
            sess.objsMan->removeObject(obj_key);
        }
        removing_objs.clear();

        zeno::GraphException::catched([&] {
            sess.mainGraph->runGraph();
        }, *sess.globalStatus);
        if (sess.globalStatus->failed()) {
            zeno::log_error(sess.globalStatus->toJson());
        }
    }

    ZENO_API void CalcManager::mark_frame_change_dirty()
    {
        getSession().mainGraph->markDirtyWhenFrameChanged();
    }

    ZENO_API void CalcManager::collect_removing_objs(std::string key)
    {
        removing_objs.insert(key);
    }

}