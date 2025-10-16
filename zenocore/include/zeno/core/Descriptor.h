#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/data.h>
#include <string>
#include <vector>

namespace zeno {

enum _ParamDescType {
    NoDesc,
    Desc_Prim,
    Desc_Obj
};

struct _ParamDesc {
    ParamObject objparam;
    ParamPrimitive primparam;
    _ParamDescType type = NoDesc;
};

struct ParamDescriptor {
  std::string name, defl, doc, comboxitems;
  size_t type;
  ParamControl control;
  _ParamDesc _desc;

  ZENO_API ParamDescriptor(size_t type,
	  std::string const &name, std::string const &defl, std::string const &doc = "", ParamControl const& controlType = NullControl);

  ZENO_API ParamDescriptor(std::string const& comboitemsDesc,
      std::string const& name,
      std::string const& defl = {});

  ZENO_API ~ParamDescriptor();
};

struct SocketDescriptor {
  std::string name, defl, doc, wildCard, comboxitems;
  size_t type = Param_Null;
  ParamControl control = NullControl;
  SocketType socketType = NoSocket;
  _ParamDesc _desc;

  ZENO_API SocketDescriptor(
      size_t type,
      std::string const &name,
      std::string const &defl = {},
      SocketType connProp = NoSocket,
      ParamControl ctrl = NullControl);

  ZENO_API SocketDescriptor(const ParamObject& param);

  ZENO_API SocketDescriptor(const ParamPrimitive& param);

  //兼容以前 `enum [items]`这种写法
  ZENO_API SocketDescriptor(
      std::string const& comboitemsDesc,
      std::string const& name,
      std::string const& defl = {}
  );
  ZENO_API ~SocketDescriptor();
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
