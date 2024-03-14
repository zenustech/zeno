#include <zeno/extra/EventCallbacks.h>
#include <zeno/extra/assetDir.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/GenericObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

#include "boost/dll.hpp"
#include "boost/dll/alias.hpp"
#include "boost/dll/import.hpp"
#include "boost/dll/shared_library.hpp"
#include "zensim/io/Filesystem.hpp"

#include "boost/make_shared.hpp"
#include "boost/shared_ptr.hpp"
#include <iostream>

namespace zeno {

struct ZenoPlugin : INode {
  void apply() override {
    auto args = has_input("args") ? get_input<DictObject>("args")
                                  : std::make_shared<DictObject>();

    auto path = get_input2<std::string>("plugin_path");
    auto name = get_input2<std::string>("node_name");

    namespace dll = boost::dll;
    namespace fs = boost::filesystem;
    try {
      dll::shared_library lib(path, dll::load_mode::append_decorations);
      auto node = dll::import <INode>(lib, name.c_str());
      node->doOnlyApply();
    } catch (const boost::system::system_error &err) {
      std::cerr << "Cannot load Plugin at " << zs::abs_exe_directory()
                << std::endl;
      std::cerr << err.what() << std::endl;
    }

    for (auto const &[k, v] : args->lut) {
      // auto handle = capiLoadObjectSharedPtr(v);
    }
    auto rets = std::make_shared<DictObject>();
#if 0
    {
      PyObject *key, *value;
      Py_ssize_t pos = 0;
      while (PyDict_Next(retsDict, &pos, &key, &value)) {
        Py_ssize_t keyLen = 0;
        const char *keyDat = PyUnicode_AsUTF8AndSize(key, &keyLen);
        if (keyDat == nullptr) {
          throw makeError("failed to cast rets key as string");
        }
        std::string keyStr(keyDat, keyLen);
        Zeno_Object handle = PyLong_AsUnsignedLongLong(value);
        if (handle == -1 && PyErr_Occurred()) {
          throw makeError("failed to cast rets value as integer");
        }
        rets->lut.emplace(std::move(keyStr), capiFindObjectSharedPtr(handle));
      }
    }
#endif
    set_output("rets", std::move(rets));
  }
};
ZENO_DEFNODE(ZenoPlugin)
({/* inputs: */
  {
      {"readpath", "plugin_path", ""},
      {"string", "node_name", ""},
      {"DictObject", "args"},
  },
  /* outputs: */
  {
      {"DictObject", "rets"},
  },
  /* params: */
  {},
  /* category: */
  {
      "plugin",
  }});

} // namespace zeno