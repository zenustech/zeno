#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/arap.h>
#include <igl/colon.h>



namespace{
using namespace zeno;


struct ArapData : zeno::IObject {
    ArapData() = default;
    igl::ARAPData data;
};

struct MakeArapSolver : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("primIn");
        auto max_iter = (int)get_param<float>("max_iter");

        if(!prim->has_attr("arap_tag")){
            throw std::runtime_error("MARK THE BOUNDARY USING ARAP_TAG");
        }

        const auto& arap_tag = prim->attr<float>("arap_tag");
        
        auto res = std::make_shared<ArapData>();
        res->data.max_iter = max_iter;

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        Eigen::VectorXi b;


        V.resize(prim->size(),3);
        F.resize(prim->quads.size(),4);
        const auto& pos = prim->attr<zeno::vec3f>("pos");
        const auto& quads = prim->quads;
        for(size_t i = 0;i < prim->size();++i)
            V.row(i) << pos[i][0],pos[i][1],pos[i][2];
        for(size_t i = 0;i < prim->quads.size();++i)
            F.row(i) << quads[i][0],quads[i][1],quads[i][2],quads[i][3];
        std::vector<int> b_vec;b_vec.clear();
        for(size_t i = 0;i < prim->size();++i)
            if(fabs(arap_tag[i] - 1.0) < 1e-6)
                b_vec.emplace_back(i);

        b.resize(b_vec.size());
        for(size_t i = 0;i < b.size();++i)
            b[i] = b_vec[i];

        igl::arap_precomputation(V,F,V.cols(),b,res->data);
        set_output("ARAP_data",std::move(res));
    }
};

ZENDEFNODE(MakeArapSolver,{
    {"primIn"},
    {"ARAP_data"},
    {{"float","max_iter","100"}},
    {"Skinning"},
});

struct DoARAPDeformation : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto arap_data = get_input<ArapData>("ARAP_data");

        Eigen::MatrixXd U;
        U.resize(prim->size(),3);
        if(!prim->has_attr("curPos")){
            throw std::runtime_error("curPos should be specified");
        }
        auto& curPos = prim->attr<zeno::vec3f>("curPos");

        Eigen::MatrixXd bc(arap_data->data.b.size(),3);

        for(size_t i = 0;i < prim->size();++i)
            U.row(i) << curPos[i][0],curPos[i][1],curPos[i][2];

        for(size_t i = 0;i < arap_data->data.b.size();++i)
            bc.row(i) = U.row(arap_data->data.b[i]);

        igl::arap_solve(bc,arap_data->data,U);

        for(size_t i = 0;i < prim->size();++i)
            curPos[i] = zeno::vec3f(U(i,0),U(i,1),U(i,2));

        set_output("prim",prim);
    }
};

ZENDEFNODE(DoARAPDeformation,{
    {"prim","ARAP_data"},
    {"prim"},
    {},
    {"Skinning"},
});


// struct TransformPrimitivePartsByTags : zeno::INode {

// };

}