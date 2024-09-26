#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/data.h>
#include <zeno/core/INode.h>
#include <zeno/core/Graph.h>
#include <reflect/core.hpp>
#include <reflect/type.hpp>
#include <reflect/metadata.hpp>
#include <reflect/registry.hpp>
#include <reflect/container/object_proxy>
#include <reflect/container/any>
#include <reflect/container/arraylist>
#include <reflect/core.hpp>
#include <zeno/core/reflectdef.h>


namespace zeno {

    struct INodeClass {
        CustomUI m_customui;
        std::string classname;

        ZENO_API INodeClass(CustomUI const& customui, std::string const& classname);
        ZENO_API virtual ~INodeClass();
        virtual std::shared_ptr<INode> new_instance(std::shared_ptr<Graph> pGraph, std::string const& classname) = 0;
    };

    struct ReflectNodeClass : INodeClass {
        std::function<std::shared_ptr<INode>()> ctor;
        zeno::reflect::TypeBase* typebase;

        ReflectNodeClass(std::function<std::shared_ptr<INode>()> ctor, std::string const& nodecls, zeno::reflect::TypeBase* pTypeBase);
        void initCustomUI();
        std::shared_ptr<INode> new_instance(std::shared_ptr<Graph> pGraph, std::string const& classname) override;
    };

    struct ImplNodeClass : INodeClass {
        std::shared_ptr<INode>(*ctor)();

        ImplNodeClass(std::shared_ptr<INode>(*ctor)(), CustomUI const& customui, std::string const& name);
        std::shared_ptr<INode> new_instance(std::shared_ptr<Graph> pGraph, std::string const& name) override;
    };
}