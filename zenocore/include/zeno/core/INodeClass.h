#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/data.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/core/Graph.h>
#include <reflect/core.hpp>
#include <reflect/type.hpp>
#include <reflect/metadata.hpp>
#include <reflect/registry.hpp>
#include <reflect/container/any>
#include <reflect/container/arraylist>
#include <zeno/core/reflectdef.h>


namespace zeno {

    struct INodeClass {
        CustomUI m_customui;
        std::string classname;

        ZENO_API INodeClass(CustomUI const& customui, std::string const& classname);
        ZENO_API virtual ~INodeClass();
        virtual std::unique_ptr<NodeImpl> new_instance(Graph* pGraph, std::string const& classname) = 0;
        virtual std::unique_ptr<INode> new_coreinst() = 0;
    };

    struct ReflectNodeClass : INodeClass {
        std::function<INode*()> ctor;
        zeno::reflect::TypeBase* typebase;

        ReflectNodeClass(std::function<INode*()> ctor, std::string const& nodecls, zeno::reflect::TypeBase* pTypeBase);
        void initCustomUI();
        std::unique_ptr<NodeImpl> new_instance(Graph* pGraph, std::string const& classname) override;
        std::unique_ptr<INode> new_coreinst() override;
    };

    struct ImplNodeClass : INodeClass {
        INode*(*ctor)();

        ImplNodeClass(INode*(*ctor)(), CustomUI const& customui, std::string const& name);
        std::unique_ptr<NodeImpl> new_instance(Graph* pGraph, std::string const& name) override;
        std::unique_ptr<INode> new_coreinst() override;
    };
}