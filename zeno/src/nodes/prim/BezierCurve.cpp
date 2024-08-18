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


static void CreateNBezierCurve(const std::vector<zeno::vec3f> src, std::vector<zeno::vec3f> &dest, double precision) {
    int size = src.size();
    std::vector<double> coff(size, 0);

    std::vector<std::vector<int>> a(size, std::vector<int>(size));
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

        zeno::vec3f ret(0, 0, 0);
        for (int i = 0; i < size; ++i) {
            zeno::vec3f tmp(src[i][0] * coff[i], src[i][1] * coff[i], src[i][2] * coff[i]);
            ret[0] = ret[0] + tmp[0];
            ret[1] = ret[1] + tmp[1];
            ret[2] = ret[2] + tmp[2];
        }
        dest.push_back(ret);
    }
}

struct CreateBezierCurve : zeno::INode {
    virtual void apply() override {
        auto precision = get_input<zeno::NumericObject>("precision")->get<float>();
        auto tag = get_param<std::string>("SampleTag");
        auto attr = get_param<std::string>("SampleAttr");

        zeno::BCurveObject outCurve;
        auto outprim = std::make_shared<zeno::PrimitiveObject>();
        std::vector<zeno::vec3f> inputPoint, cPoints;
        std::vector<float> tagList;
       
        if (has_input<zeno::ListObject>("CustomPoints")) {        
            auto list = get_input<zeno::ListObject>("CustomPoints");
            int iSize = list->size();
            if (iSize > 0) {
                for (int i = 0; i < iSize; i++) {
                    zeno::PrimitiveObject *obj = dynamic_cast<zeno::PrimitiveObject *>(list->get(i).get());
                    for (auto p : obj->verts) {
                        inputPoint.push_back(p);
                    }
                }
            } 
        } else {
            auto inPrim = get_input<zeno::PrimitiveObject>("SamplePoints").get();
            if (attr.empty()) {
                set_output("prim", std::move(std::shared_ptr<zeno::PrimitiveObject>(new zeno::PrimitiveObject)));
                set_primitive_output("curev", outCurve);
                return;
            }
            auto tmpPs = inPrim->attr<zeno::vec3f>(attr);
            for (auto p : tmpPs) {
                inputPoint.push_back(p);
            }

            auto tmpTags = inPrim->attr<float>(tag);
            for (auto p : tmpTags) {
                tagList.push_back(p);
            }
        }

        CreateNBezierCurve(inputPoint, cPoints, precision);

        zeno::log_info("input point size: {}", inputPoint.size());
        zeno::log_info("output point size: {}", cPoints.size());
        zeno::log_info("precision : {}", precision);
        
        outCurve.points = inputPoint;
        outCurve.precision = precision;
        outCurve.bPoints = cPoints;
        outCurve.sampleTag = tag;

        if (tagList.size() > 0)
        {
            auto tmpScal = 1 / precision;
            outprim->verts.resize(tagList.size());
            for (int i = 0; i < tagList.size(); i++) {
                int idx = tagList.at(i) * tmpScal;
                outprim->verts[i] = cPoints[idx];
            }
        } else {
            outprim->verts.resize(cPoints.size());
            for (int i = 0; i < cPoints.size(); i++) {
                outprim->verts[i] = cPoints[i];
            }
        }       
        set_output("prim", std::move(outprim));
        set_primitive_output("curev", outCurve);
    }
};

ZENDEFNODE(CreateBezierCurve, {{
                                   {gParamType_List, "CustomPoints"}, //input
                                   {gParamType_Primitive, "SamplePoints"}, //input
                                   {gParamType_Float, "precision", "0.01"},
                               },
                               {gParamType_Primitive,{gParamType_Curve, "curev"}}, //output
                               {
                                   {"enum Bezier", "Type", "Bezier"},
                                   {gParamType_String, "SampleTag", ""},
                                   {gParamType_String, "SampleAttr", ""},
                               },           //prim
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
                         {{gParamType_Primitive, "prim"}},
                         /*参数*/
                         {
                             {gParamType_Float, "x", "0"},
                             {gParamType_Float, "y", "0"},
                             {gParamType_Float, "z", "0"},
                         },
                         /*类别*/
                         {"create"}});
