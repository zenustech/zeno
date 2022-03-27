#include "Structures.hpp"
#include "Utils.hpp"
#include <cassert>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/container/Vector.hpp"

namespace zeno {

struct ZSLinkTest : INode {
	void apply() override {
        using namespace zs;
        zs::initialize_openvdb();
        zs::Vector<int> a{ 100, memsrc_e::um, 0 };
        a[0] = 100;
        fmt::print("first element: {}\n", a[0]);
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
