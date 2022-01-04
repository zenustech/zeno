#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>


#include <igl/readDMAT.h>
#include <igl/column_to_quats.h>

#include "skinning_iobject.h"

namespace{
using namespace zeno;

struct ReadPoseFrame : zeno::INode {
    virtual void apply() override {
        auto res = std::make_shared<PosesAnimationFrame>();
        auto bones = get_input<zeno::PrimitiveObject>("bones");
        auto dmat_path = get_input<zeno::StringObject>("dmat_path")->get();

        Eigen::MatrixXd Q;
        igl::readDMAT(dmat_path,Q);

        if(bones->lines.size() != Q.rows()/4 || Q.rows() % 4 != 0){
            std::cout << "THE DIMENSION OF BONES DOES NOT MATCH POSES  " << bones->lines.size() << "\t" << Q.rows() << std::endl;
            throw std::runtime_error("THE DIMENSION OF BONES DOES NOT MATCH POSES");
        }

        igl::column_to_quats(Q,res->posesFrame);
        set_output("posesFrame",std::move(res));
    }
};

ZENDEFNODE(ReadPoseFrame, {
    {{"readpath","dmat_path"},"bones"},
    {"posesFrame"},
    {},
    {"Skinning"},
});

}