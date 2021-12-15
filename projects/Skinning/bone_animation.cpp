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

struct MakeRestPoses : zeno::INode {
    virtual void apply() override {
        auto res = std::make_shared<PosesAnimationFrame>();
        auto bones = get_input<zeno::PrimitiveObject>("bones");

        res->posesFrame.resize(bones->lines.size(),Eigen::Quaterniond::Identity());

        set_output("posesFrame",std::move(res));
    }
};

ZENDEFNODE(MakeRestPoses, {
    {"bones"},
    {"posesFrame"},
    {},
    {"Skinning"},
});


struct BlendPoses : zeno::INode {
    virtual void apply() override {
        auto poses1 = get_input<PosesAnimationFrame>("p1");
        auto poses2 = get_input<PosesAnimationFrame>("p2");

        // if(poses1->posesFrame.cols() != 4 || poses1->posesFrame.cols() != 4)
        //     throw std::runtime_error("INVALIED POSES DIMENSION");

        if(poses1->posesFrame.size() != poses2->posesFrame.size()){
            std::cout << "THE DIMENSION OF TWO MERGED POSES DOES NOT MATCH " << poses1->posesFrame.size() << "\t" << poses2->posesFrame.size() << std::endl;
            throw std::runtime_error("THE DIMENSION OF TWO MERGED POSES DOES NOT MATCH");
        }

        auto w = get_input<zeno::NumericObject>("w")->get<float>();

        auto res = std::make_shared<PosesAnimationFrame>();
        res->posesFrame.resize(poses1->posesFrame.size());

        for(size_t i = 0;i < res->posesFrame.size();++i){
            res->posesFrame[i] =  poses1->posesFrame[i].slerp(1-w,poses2->posesFrame[i]);
        }

        std::cout << "OUT_POSES : " << std::endl;
        for(size_t i = 0;i < res->posesFrame.size();++i)
            std::cout << "P<" << i << "> : " << res->posesFrame[i] << std::endl;

        set_output("bp",std::move(res));
    }
};

ZENDEFNODE(BlendPoses, {
    {"p1","p2","w"},
    {"bp"},
    {},
    {"Skinning"},
});

}