#pragma once

#include <zeno/core/NodeImpl.h>


namespace zeno
{
    class SolverImpl : public NodeImpl
    {
    public:
        SolverImpl(INode* pNode);
        virtual ~SolverImpl();
        void terminate_solve();
        IObject* get_default_output_object() override;
        void clearCalcResults() override;
        void dirty_changed(bool bOn, DirtyReason reason, bool bWholeSubnet, bool bRecursively) override;
    };
}