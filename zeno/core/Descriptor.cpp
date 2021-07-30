#include <zeno/core/Descriptor.h>
#include <zeno/utils/string.h>

namespace zeno {

SocketDescriptor::SocketDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl)
      : type(type), name(name), defl(defl) {}
SocketDescriptor::~SocketDescriptor() = default;


ParamDescriptor::ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl)
      : type(type), name(name), defl(defl) {}
ParamDescriptor::~ParamDescriptor() = default;

ZENO_API Descriptor::Descriptor() = default;
ZENO_API Descriptor::Descriptor(
  std::vector<SocketDescriptor> const &inputs,
  std::vector<SocketDescriptor> const &outputs,
  std::vector<ParamDescriptor> const &params,
  std::vector<std::string> const &categories)
  : inputs(inputs), outputs(outputs), params(params), categories(categories) {
    this->inputs.push_back("SRC");
    //this->inputs.push_back("COND");  // deprecated
    this->outputs.push_back("DST");
}

ZENO_API std::string Descriptor::serialize() const {
  std::string res = "";
  std::vector<std::string> strs;
  for (auto const &[type, name, defl] : inputs) {
      strs.push_back(type + "@" + name + "@" + defl);
  }
  res += "{" + join_str(strs, "%") + "}";
  strs.clear();
  for (auto const &[type, name, defl] : outputs) {
      strs.push_back(type + "@" + name + "@" + defl);
  }
  res += "{" + join_str(strs, "%") + "}";
  strs.clear();
  for (auto const &[type, name, defl] : params) {
      strs.push_back(type + "@" + name + "@" + defl);
  }
  res += "{" + join_str(strs, "%") + "}";
  res += "{" + join_str(categories, "%") + "}";
  return res;
}

}
