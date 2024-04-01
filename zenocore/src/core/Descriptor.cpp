#include <zeno/core/Descriptor.h>

namespace zeno {

SocketDescriptor::SocketDescriptor(
        std::string const &type,
        std::string const &name,
        std::string const &defl,
        SocketType connProp,
        ParamControl ctrl,
        std::string const &doc)
        : type(type)
        , name(name)
        , defl(defl)
        , doc(doc)
        , socketType(connProp)
        , control(ctrl) {}

SocketDescriptor::~SocketDescriptor() = default;


ParamDescriptor::ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl, std::string const &doc)
      : type(type), name(name), defl(defl), doc(doc) {}
ParamDescriptor::~ParamDescriptor() = default;

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
