#include <zeno/core/SolverImpl.h>
#include <zeno/extra/flipsolver.h>


namespace zeno
{
    SolverImpl::SolverImpl(INode* pNode) : NodeImpl(pNode) {

    }

    SolverImpl::~SolverImpl() {

    }

    void SolverImpl::terminate_solve() {
        if (get_nodecls() == "FlipSolver") {
            FlipSolver* pNode = static_cast<FlipSolver*>(coreNode());
            pNode->terminate_solve();
        }
    }

    void SolverImpl::clearCalcResults() {
        NodeImpl::clearCalcResults();
        if (get_nodecls() == "FlipSolver") {
            FlipSolver* pNode = static_cast<FlipSolver*>(coreNode());
            pNode->objs_cleaned();
        }
    }

    void SolverImpl::dirty_changed(bool bOn, DirtyReason reason, bool bWholeSubnet, bool bRecursively) {
        if (get_nodecls() == "FlipSolver") {
            FlipSolver* pNode = static_cast<FlipSolver*>(coreNode());
            pNode->on_dirty_changed(bOn, reason, bWholeSubnet, bRecursively);
        }
    }

    IObject* SolverImpl::get_default_output_object() {
        if (get_nodecls() == "FlipSolver") {
            FlipSolver* pNode = static_cast<FlipSolver*>(coreNode());
            //这里取到的帧未必就是当前ui的帧，因为用户可能不断滑动时间轴
            int frame = zeno::getSession().globalState->getFrameId();
            std::string cache_path;
            if (has_input("Cache Path")) {
                cache_path = get_input_prim<std::string>("Cache Path");
            }
            zany result = pNode->checkCache(cache_path, frame);
            if (result && result->key().empty()) {
                result->update_key(stdString2zs(get_uuid()));
            }
            return result.release();
        }
        return nullptr;
    }
}