#pragma once

#include <zeno/utils/api.h>
#include <string>
#include <vector>

namespace zeno {

struct ParamDescriptor {
  std::string type, name, defl;

  ZENO_API ParamDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl);
  ZENO_API ~ParamDescriptor();
};

struct SocketDescriptor {
  std::string type, name, defl;

  ZENO_API SocketDescriptor(std::string const &type,
	  std::string const &name, std::string const &defl = {});
  ZENO_API ~SocketDescriptor();

  //[[deprecated("use {\"sockType\", \"sockName\"} instead of \"sockName\"")]]
  SocketDescriptor(const char *name)
      : SocketDescriptor({}, name) {}
};

struct Descriptor {
  std::vector<SocketDescriptor> inputs;
  std::vector<SocketDescriptor> outputs;
  std::vector<ParamDescriptor> params;
  std::vector<std::string> categories;

  ZENO_API Descriptor();
  ZENO_API Descriptor(
	  std::vector<SocketDescriptor> const &inputs,
	  std::vector<SocketDescriptor> const &outputs,
	  std::vector<ParamDescriptor> const &params,
	  std::vector<std::string> const &categories);

  ZENO_API std::string serialize() const;
};

}
