#include <zeno/core/Descriptor.h>
#include "zeno_types/reflect/reflection.generated.hpp"

namespace zeno {

    SocketDescriptor::SocketDescriptor(
        size_t type,
        std::string const &name,
        std::string const &defl,
        SocketType connProp,
        ParamControl ctrl,
        std::string const& wildCard,
        std::string const& doc,
        std::string const& cboxitems)
        : type(type)
        , name(name)
        , defl(defl)
        , doc(doc)
        , wildCard(wildCard)
        , socketType(connProp)
        , comboxitems(cboxitems)
        , control(ctrl) {}

SocketDescriptor::SocketDescriptor(std::string const& comboitemsDesc, std::string const& name, std::string const& defl)
    : type(zeno::types::gParamType_String)
    , name(name)
    , socketType(zeno::Socket_Primitve)
    , control(zeno::Combobox)
    , comboxitems(comboitemsDesc)
    , defl(defl)
{
}

SocketDescriptor::~SocketDescriptor() = default;


ParamDescriptor::ParamDescriptor(size_t type,
	  std::string const &name, std::string const &defl, std::string const &doc)
      : type(type), name(name), defl(defl), doc(doc) {}
ParamDescriptor::~ParamDescriptor() = default;

ParamDescriptor::ParamDescriptor(std::string const& comboitemsDesc, std::string const& name, std::string const& defl)
    : type(zeno::types::gParamType_String)
    , name(name)
    , comboxitems(comboitemsDesc)
    , defl(defl)
{
}

ZENO_API Descriptor::Descriptor() = default;
ZENO_API Descriptor::Descriptor(
  std::vector<SocketDescriptor> const &inputs,
  std::vector<SocketDescriptor> const &outputs,
  std::vector<ParamDescriptor> const &params,
  std::vector<std::string> const &categories,
  std::string const &displayName,
  std::string const &iconResPath,
  std::string const& doc)
  : inputs(inputs)
  , outputs(outputs)
  , params(params)
  , categories(categories)
  , doc(doc)
  , displayName(displayName)
  , iconResPath(iconResPath) {
    //this->inputs.push_back("SRC");
    //this->inputs.push_back("COND");  // deprecated
    //this->outputs.push_back("DST");
}

}
