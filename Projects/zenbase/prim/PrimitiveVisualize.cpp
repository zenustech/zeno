#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/Visualization.h>

virtual void PrimitiveObject::visualize() {
    auto path = Visualization::exportPath("zpm");
    printf("vis path is %s\n", path.c_str());
}
