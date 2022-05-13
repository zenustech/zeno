#include <zeno/zeno.h>
#include <zeno/utils/logger.h>

struct EvalAnim{

};

struct EvalFBXAnim : zeno::INode {

    virtual void apply() override {

    }
};
ZENDEFNODE(EvalFBXAnim,
           {       /* inputs: */
               {
                   {"frameid"}
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

