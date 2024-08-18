#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/data.h>
#include <string>
#include <vector>

namespace zeno {

struct ParamDescriptor {
  std::string name, defl, doc, comboxitems;
  size_t type;

  ZENO_API ParamDescriptor(size_t type,
	  std::string const &name, std::string const &defl, std::string const &doc = "");

  ZENO_API ParamDescriptor(std::string const& comboitemsDesc,
      std::string const& name,
      std::string const& defl = {});

  ZENO_API ~ParamDescriptor();
};

struct SocketDescriptor {
  std::string name, defl, doc, wildCard, comboxitems;
  size_t type;
  ParamControl control;
  SocketType socketType;

  ZENO_API SocketDescriptor(
      size_t type,
      std::string const &name,
      std::string const &defl = {},
      SocketType connProp = NoSocket,
      ParamControl ctrl = NullControl,
      std::string const&wildCard = {},
      std::string const &doc = {});

  //兼容以前 `enum [items]`这种写法
  ZENO_API SocketDescriptor(
      std::string const& comboitemsDesc,
      std::string const& name,
      std::string const& defl = {}
  );
  ZENO_API ~SocketDescriptor();

  //[[deprecated("use {\"sockType\", \"sockName\"} instead of \"sockName\"")]]
  //SocketDescriptor(const char *name)
  //    : SocketDescriptor({}, name) {}	
};

struct Descriptor {
  std::vector<SocketDescriptor> inputs;
  std::vector<SocketDescriptor> outputs;
  std::vector<ParamDescriptor> params;
  std::vector<std::string> categories;
  std::string doc;
  std::string displayName;
  std::string iconResPath;

  ZENO_API Descriptor();
  ZENO_API Descriptor(
      std::vector<SocketDescriptor> const &inputs,
      std::vector<SocketDescriptor> const &outputs,
      std::vector<ParamDescriptor> const &params,
      std::vector<std::string> const &categories,
      std::string const &displayName = "",
      std::string const &iconResPath = "",
      std::string const& doc = "");
};

}
