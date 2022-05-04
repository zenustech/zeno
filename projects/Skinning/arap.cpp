#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/arap.h>
#include <igl/colon.h>


#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/ext/matrix_transform.hpp>


namespace{
using namespace zeno;


struct ArapData : zeno::IObject {
    ArapData() = default;
    igl::ARAPData data;
};

struct MakeArapSolver : zeno::INode {
    virtual void apply() override {
        std::cout << "MAKE ARAP SOLVER" << std::endl;

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


        #pragma omp parallel for 
        for(intptr_t i = 0;i < prim->size();++i)
            U.row(i) << curPos[i][0],curPos[i][1],curPos[i][2];

        #pragma omp parallel for 
        for(intptr_t i = 0;i < arap_data->data.b.size();++i)
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


struct TransformPrimitivePartsByTags : zeno::INode {

    static glm::vec3 mapplypos(glm::mat4 const &matrix, glm::vec3 const &vector) {
        auto vector4 = matrix * glm::vec4(vector, 1.0f);
        return glm::vec3(vector4) / vector4.w;
    }

    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        const auto& Qs = get_input<zeno::ListObject>("Qs")->get<std::shared_ptr<zeno::NumericObject>>();
        const auto& Ts = get_input<zeno::ListObject>("Ts")->get<std::shared_ptr<zeno::NumericObject>>();
        const auto& tagName = get_input2<std::string>("tagName");

        // std::cout << "HERE" << std::endl;

        size_t nm_tags = Qs.size();
        if(Ts.size() != nm_tags)
            throw std::runtime_error("input Qs and Ts should have same size");

        if(!prim->has_attr("curPos"))
            throw std::runtime_error("input primitive does not has curPos channel");

         if(!prim->has_attr(tagName))
            throw std::runtime_error("input primitive does not has specified tag channel");


        std::vector<glm::mat4> As(nm_tags);

        // std::cout << "AS : " << std::endl;

        // for(size_t i = 0;i < nm_tags;++i){
        //     const auto& T = Ts[i]->get<zeno::vec3f>();
        //     const auto& Q = Qs[i]->get<zeno::vec4f>();
        //     glm::mat4 matTrans = glm::translate(glm::vec3(T[0],T[1],T[2]));
        //     glm::quat matQuat(Q[3],Q[0],Q[1],Q[2]);
        //     glm::mat4 matRot = glm::toMat4(matQuat);

        //     As[i] = matTrans * matRot;


        //     // std::cout   << As[i][0][0] << "\t" << As[i][0][1] << "\t" << As[i][0][2] << "\t" << As[i][0][3] << std::endl \
        //     //             << As[i][1][0] << "\t" << As[i][1][1] << "\t" << As[i][1][2] << "\t" << As[i][1][3] << std::endl \
        //     //             << As[i][2][0] << "\t" << As[i][2][1] << "\t" << As[i][2][2] << "\t" << As[i][2][3] << std::endl \
        //     //             << As[i][3][0] << "\t" << As[i][3][1] << "\t" << As[i][3][2] << "\t" << As[i][3][3] << std::endl;

        // }

        const auto& pos = prim->attr<zeno::vec3f>("pos");
        auto& curPos = prim->attr<zeno::vec3f>("curPos");


        const auto& tags = prim->attr<float>(tagName);

        #pragma omp parallel for 
        for(intptr_t i = 0;i < prim->size();++i){
            auto tag = tags[i];
            if(tag > -1e-6)
                // std::cout << "V<" << i << ">\t: " << tag << std::endl;
            for(size_t j = 0;j < nm_tags;++j){
                if(fabs(tag - (float)j) < 1e-6){
                    // std::cout << "ALTER VERTEX" << std::endl;
                    auto p = zeno::vec_to_other<glm::vec3>(pos[i]);
                    p = mapplypos(As[j], p);
                    curPos[i] = zeno::other_to_vec<3>(p);
                }
            }
        }

        set_output("outPrim",prim);
    }
};

ZENDEFNODE(TransformPrimitivePartsByTags,{
    {"prim","Qs","Ts",{"string","tagName","RENAME_ME"}},
    {"outPrim"},
    {},
    {"Skinning"},
});

}
