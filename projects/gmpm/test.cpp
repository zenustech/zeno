#include "Structures.hpp"
#include "Utils.hpp"
#include <cassert>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>

namespace zeno {

struct ZSLinkTest : INode {
	void apply() override {
		fmt::print("loaded!\n");
		getchar();
	}
};

ZENDEFNODE(ZSLinkTest,
           {
               {},
               {},
               {},
               {"ZPCTest"},
           });

} // namespace zeno
