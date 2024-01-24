#include <zeno/core/Descriptor.h>

namespace zeno {

SocketDescriptor::SocketDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl, std::string const &doc)
      : type(type), name(name), defl(defl), doc(doc) {}
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
  std::string const &doc)
  : inputs(inputs), outputs(outputs), params(params), categories(categories), doc(doc) {
    //this->inputs.push_back("SRC");
    //this->inputs.push_back("COND");  // deprecated
    //this->outputs.push_back("DST");
}

}
