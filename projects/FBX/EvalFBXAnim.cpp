#include <zeno/zeno.h>
#include <zeno/utils/logger.h>


struct EvalFBXAnim : zeno::INode {

    virtual void apply() override {

    }
};
ZENDEFNODE(EvalFBXAnim,
           {       /* inputs: */
               {
                   {"prim"},
                   {"bone"}
               },  /* outputs: */
               {
                   "prim",
               },  /* params: */
               {

               },  /* category: */
               {
                   "primitive",
               }
           });

