#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/logger.h>
#include <zeno/zeno.h>
#include <vector>


/**
 * @brief createNBezierCurve 生成N阶贝塞尔曲线点
 * @param src 源贝塞尔控制点
 * @param dest 目的贝塞尔曲线点
 * @param precision 生成精度
 */
static void CreateNBezierCurve(const std::vector<zeno::vec2f> src, std::vector<zeno::vec2f> &dest, double precision) {
    int size = src.size();
    std::vector<double> coff(size, 0);

    std::vector<std::vector<int>> a(size, std::vector<int>(size)); // javabean
    {
        for(int i=0;i<size;++i)
        {
            a[i][0]=1;
            a[i][i]=1;
        }        
        for(int i=1;i<size;++i)
            for(int j=1;j<i;++j)
                a[i][j] = a[i-1][j-1] + a[i-1][j];

    }

    for (double t1 = 0; t1 < 1; t1 += precision) {
        double t2  = 1 - t1;
        int n = size - 1;

        coff[0] = pow(t2, n);
        coff[n] = pow(t1, n);
        for (int i = 1; i < size - 1; ++i) {
            coff[i] = pow(t2, n - i) * pow(t1, i) * a[n][i];
        }

        zeno::vec2f ret(0, 0);
        for (int i = 0; i < size; ++i) {
            zeno::vec2f tmp(src[i][0] * coff[i], src[i][1] * coff[i]);
            ret[0] = ret[0] + tmp[0];
            ret[1] = ret[1] + tmp[1];
        }
        dest.push_back(ret);
    }
}

struct CreateBezierCurve : zeno::INode {
    virtual void apply() override {
        auto list = get_input<zeno::ListObject>("Points");
        auto precision = get_input<zeno::NumericObject>("precision")->get<float>();
        int iSize = list->arr.size();
        std::vector<zeno::vec2f> inputPoint;
        for (int i = 0; i < iSize; i++) {
            zeno::PrimitiveObject *obj = dynamic_cast<zeno::PrimitiveObject *>(list->arr[i].get());
            for (auto p : obj->verts) {
                inputPoint.push_back({p[0], p[1]});
                zeno::log_info("input point: {} {}", p[0], p[1]);
            }                
        }

        std::vector<zeno::vec2f> cPoints;
        CreateNBezierCurve(inputPoint, cPoints, precision);
        
        zeno::log_info("input point size: {}", inputPoint.size());
        zeno::log_info("output point size: {}", cPoints.size());
        zeno::log_info("precision : {}", precision);
        auto outprim = new zeno::PrimitiveObject;
        outprim->verts.resize(cPoints.size());
        for (int i = 0; i < cPoints.size(); i++) {
            outprim->verts[i] = {cPoints[i][0], cPoints[i][1], 0};
        }
        set_output("prim", std::move(std::shared_ptr<zeno::PrimitiveObject>(outprim)));
    }
};

ZENDEFNODE(CreateBezierCurve, {{
                                   {"list", "Points"}, //input
                                   {"float", "precision"},
                               },
                               {"prim"},     //output
                               {},           //prim
                               {"create"}}); //cate

struct CreatePoint : zeno::INode {

    virtual void apply() override {
        //auto point = get_input<zeno::NumericObject>("Point")->get<zeno::vec2f>();
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        auto z = get_param<float>("z");
        auto outprim = std::make_shared<zeno::PrimitiveObject>();
        outprim->verts.resize(1);
        outprim->verts[0] = zeno::vec3f(x, y, z);

        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(CreatePoint, {/*输入*/
                         {

                         },
                         /*输出*/
                         {"prim"},
                         /*参数*/
                         {
                             {"float", "x", "0"},
                             {"float", "y", "0"},
                             {"float", "z", "0"},
                         },
                         /*类别*/
                         {"create"}});
