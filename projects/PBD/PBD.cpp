#include <zeno\zeno.h>
#include "PBD.h"

namespace zeno {
    struct PBD : zeno::INode {
        virtual void apply() override {
            
        }
    };
    ZENDEFNODE(PBD, {
         {
            {}
         },
         {
         },
         {},
         {
             "PBD"
         },
    });


} // namespace zeno
