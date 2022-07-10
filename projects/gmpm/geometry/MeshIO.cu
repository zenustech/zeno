#include "file_parser/read_vtk_mesh.hpp"

#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {
    struct ReadVTKMesh : INode {
        void apply() override {
            auto path = get_input<StringObject>("path")->get();

        }
    }
}