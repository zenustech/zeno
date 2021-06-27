#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/Visualization.h>
#include <zen/PrimitiveIO.h>

namespace zen {

void PrimitiveObject::visualize() {
    auto path = Visualization::exportPath("zpm");
    writezpm(this, path.c_str());
}

}
