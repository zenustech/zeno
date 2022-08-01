#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/gradient_field.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/math/MathUtils.h"


namespace zeno {

struct ZSEvalGradientField : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;

    virtual void apply() override {
        using namespace zs;
        auto field = get_input<ZenoParticles>("field");
        auto& verts = field->getParticles();

        auto attr = get_param<std::string>("tag");
        auto attrg = get_param<std::string>("gtag");
        auto skip_boundary_gradient = get_param<int>("skip_boundary");
        auto btag = get_param<std::string>("btag");
        auto normalize = get_param<int>("normalize");

        if(!verts.hasProperty(attr)){
            fmt::print("the input field does not contain specified channel:{}\n",attr);
            throw std::runtime_error("the input field does not contain specified channel");
        }
        if(verts.getChannelSize(attr) != 1){
            fmt::print("only scaler field is currently supported\n");
            throw std::runtime_error("only scaler field is currently supported");
        }

        auto& eles = field->getQuadraturePoints();
        auto simplex_size = eles.getChannelSize("inds");
        if(simplex_size != 4 && simplex_size != 3)
            throw std::runtime_error("ZSEvalGradientField: invalid simplex size");

        static dtiles_t etemp(eles.get_allocator(),{{"g",3}},eles.size());
        static dtiles_t vtemp{verts.get_allocator(),{
            {"T",1},
        },verts.size()};

        etemp.resize(eles.size());
        vtemp.resize(verts.size());

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        eles.append_channels(cudaPol,{{attrg,3}});
        // copy the scaler field from verts to vtemp
        cudaPol(zs::range(verts.size()),
            [verts = proxy<space>({},verts),vtemp = proxy<space>({},vtemp),attr = zs::SmallString(attr),tag = zs::SmallString("T")]
                ZS_LAMBDA(int vi) mutable {
                    vtemp(tag,vi) = verts(attr,vi);
        });
        // compute_gradient(cudaPol,eles,verts,"x",vtemp,"T",etemp,"g",zs::wrapv(simplex_size));
        if(simplex_size == 4)
            compute_gradient<4>(cudaPol,eles,verts,"x",vtemp,"T",etemp,"g");
        if(simplex_size == 3)
            compute_gradient<3>(cudaPol,eles,verts,"x",vtemp,"T",etemp,"g");
        // copy the gradient field from etemp to eles
        cudaPol(zs::range(eles.size()),
            [verts = proxy<space>({},verts),eles = proxy<space>({},eles),etemp = proxy<space>({},etemp),gtag = zs::SmallString(attrg),normalize,skip_boundary_gradient,btag = zs::SmallString(btag),simplex_size]
                ZS_LAMBDA(int ei) mutable {
                    bool on_bou = false;
                    if(simplex_size == 3){
                        auto inds = eles.template pack<3>("inds",ei).reinterpret_bits<int>();
                        for(int i = 0;i < simplex_size;++i){
                            auto b = verts(btag,inds[i]);
                            if(b > 1e-6)
                                on_bou = true;
                        }
                    }
                    if(simplex_size == 4){
                        auto inds = eles.template pack<4>("inds",ei).reinterpret_bits<int>();
                        for(int i = 0;i < simplex_size;++i){
                            auto b = verts(btag,inds[i]);
                            if(b > 1e-6)
                                on_bou = true;
                        }
                    }

                    if(on_bou && skip_boundary_gradient){
                        eles.tuple<3>(gtag,ei) = vec3::zeros();
                    }else {
                        float alpha = 1;
                        if(normalize)
                            alpha = (etemp.template pack<3>("g",ei).norm());
                        eles.template tuple<3>(gtag,ei) = etemp.template pack<3>("g",ei)/alpha;

                    }
        });
        set_output("field",field);
    }
};

ZENDEFNODE(ZSEvalGradientField, {
                                    {"field"},
                                    {"field"},
                                    {
                                        {"string","tag","T"},{"string","gtag","gradT"},{"int","skip_boundary","0"},{"int","normalize","0"},{"string","btag","btag"}
                                    },
                                    {"ZSGeometry"}
});

struct ZSRetrieveVectorField : zeno::INode {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T,3>;
    using mat3 = zs::vec<T,3,3>;
    virtual void apply() override {
        using namespace zs;
        auto field = get_input<ZenoParticles>("field");
        const auto& verts = field->getParticles();
        const auto& eles = field->getQuadraturePoints(); 

        auto type = get_param<std::string>("location");
        auto gtag = get_param<std::string>("gtag");
        auto xtag = get_param<std::string>("xtag");
        // auto normalize = get_param<int>("normalize");
        auto scale = (T)get_param<float>("scale");

        if(type == "element" && !eles.hasProperty(gtag)){
            fmt::print("the volume does not contain element-wise gradient field : {}\n",gtag);
            throw std::runtime_error("the volume does not contain element-wise gradient field");
        }
        if(type == "vert" && !verts.hasProperty(gtag)){
            fmt::print("the volume does not contain nodal-wize gradient field : {}\n",gtag);
            throw std::runtime_error("the volume does not contain nodal-wize gradient field");
        }
        if(!verts.hasProperty(xtag)){
            fmt::print("the volume does not contain specified position channel : {}\n",xtag);
            throw std::runtime_error("the volume does not contain specified position channel");
        }

        std::vector<zs::PropertyTag> tags{{"x",3},{"vec",3}/*,{"clr",3}*/};
        bool on_elm = (type == "element");
        auto vec_buffer = typename ZenoParticles::particles_t(tags,on_elm ? eles.size() : verts.size(),zs::memsrc_e::device,0);
        // transfer the data from gpu to cpu
        constexpr auto cuda_space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        int elm_dim = eles.getChannelSize("inds");

        cudaPol(zs::range(vec_buffer.size()),
            [vec_buffer = proxy<cuda_space>({},vec_buffer),verts = proxy<cuda_space>({},verts),eles = proxy<cuda_space>({},eles),
                gtag = zs::SmallString(gtag),xtag = zs::SmallString(xtag),on_elm,scale,elm_dim] ZS_LAMBDA(int i) mutable {
                    if(on_elm){
                        auto bx = vec3::zeros();
                        if(elm_dim == 4){
                            auto inds = eles.pack<4>("inds",i).reinterpret_bits<int>();
                            for(int j = 0;j != 4;++j)
                                bx += verts.pack<3>(xtag,inds[j]) / 4;
                        }else if(elm_dim == 3){
                            auto inds = eles.pack<3>("inds",i).reinterpret_bits<int>();
                            for(int j= 0;j != 3;++j)
                                bx += verts.pack<3>("xtag",inds[j]) / 3;
                        }
                        vec_buffer.tuple<3>("x",i) = bx;
                        vec_buffer.tuple<3>("vec",i) = scale * eles.pack<3>(gtag,i)/* / eles.pack<3>(gtag,i).norm()*/;
                    }else{
                        vec_buffer.tuple<3>("x",i) = verts.pack<3>(xtag,i);
                        vec_buffer.tuple<3>("vec",i) = scale * verts.pack<3>(gtag,i)/* / verts.pack<3>(gtag,i).norm()*/;
                    }
        });

        vec_buffer = vec_buffer.clone({zs::memsrc_e::host});
        int vec_size = vec_buffer.size();
        constexpr auto omp_space = execspace_e::openmp;
        auto ompPol = omp_exec();

        auto vec_field = std::make_shared<zeno::PrimitiveObject>();
        vec_field->resize(vec_size * 2);
        auto& segs = vec_field->lines;
        segs.resize(vec_size);
        auto& sverts = vec_field->attr<zeno::vec3f>("pos");

        ompPol(zs::range(vec_buffer.size()),
            [vec_buffer = proxy<omp_space>({},vec_buffer),&segs,&sverts,vec_size] (int i) mutable {
                segs[i] = zeno::vec2i(i * 2 + 0,i * 2 + 1);
                auto start = vec_buffer.pack<3>("x",i);
                auto end = start + vec_buffer.pack<3>("vec",i);
                sverts[i*2 + 0] = zeno::vec3f{start[0],start[1],start[2]};
                sverts[i*2 + 1] = zeno::vec3f{end[0],end[1],end[2]};
        });

        set_output("vec_field",std::move(vec_field));
    }    
};

ZENDEFNODE(ZSRetrieveVectorField, {
    {"field"},
    {"vec_field"},
    {{"enum element vert","location","element"},{"string","gtag","vec_field"},{"string","xtag","xtag"},{"float","scale","1.0"}},
    {"ZSGeometry"},
});


struct ZSSampleQuadratureAttr2Vert : zeno::INode {
    void apply() override {
        using namespace zs;
        auto field = get_input<ZenoParticles>("ZSParticles");
        auto& verts = field->getParticles();
        auto& quads = field->getQuadraturePoints();
    
        auto attr = get_param<std::string>("attr");
        auto weight = get_param<std::string>("wtag");

        auto skip_bou = get_param<int>("skip_bou");
        auto bou_tag = get_param<std::string>("bou_tag");

        if(!quads.hasProperty(attr)){
            fmt::print("the input quadrature does not have specified attribute : {}\n",attr);
            throw std::runtime_error("the input quadrature does not have specified attribute");
        }

        if(!quads.hasProperty(weight)){
            fmt::print("the input quadratures does not have specified weight attribute : {}\n",weight);
            throw std::runtime_error("the input quadratures does not have specified weight attribute");
        }

        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        int attr_dim = quads.getChannelSize(attr);
        int simplex_size = quads.getChannelSize("inds");

        verts.append_channels(cudaPol,{{attr,attr_dim}});
        cudaPol(range(verts.size()),
            [verts = proxy<space>({},verts),attr_dim,attr = SmallString(attr)] 
                __device__(int vi) mutable {
                    for(int i = 0;i != attr_dim;++i)
                        verts(attr,i,vi) = 0.;
        });

        cudaPol(range(quads.size()),
            [verts = proxy<space>({},verts),quads = proxy<space>({},quads),attr_dim,attr = SmallString(attr),simplex_size,weight = SmallString(weight),execTag = wrapv<space>{},skip_bou,bou_tag = zs::SmallString(bou_tag)]
                __device__(int ei) mutable {
                    float w = quads(weight,ei);
                    // w = 1.0;// cancel out the specified weight info
                    for(int i = 0;i != simplex_size;++i){
                        auto idx = reinterpret_bits<int>(quads("inds",i,ei));
                        auto bou = verts(bou_tag,idx);
                        if(bou && skip_bou)
                            continue;
                        for(int j = 0;j != attr_dim;++j) {
                            // verts(attr,j,idx) += w * quads(attr,j,ei) / (float)simplex_size;
                            auto alpha = w * quads(attr,j,ei) / (float)simplex_size;
                            atomic_add(execTag,&verts(attr,j,idx),alpha);
                        }
                    }                    
        });

        set_output("ZSParticles",field);
    }
};

ZENDEFNODE(ZSSampleQuadratureAttr2Vert,{
    {"ZSParticles"},
    {"ZSParticles"},
    {
        {"string","attr","attr"},{"string","wtag","vol"},{"int","skip_bou","0"},{"string","bou_tag","btag"}
    },
    {"ZSGeometry"}
});


struct ZSSampleVertAttr2Quadrature : zeno::INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto field = get_input<ZenoParticles>("field");
        auto& verts = field->getParticles();
        auto& quads = field->getQuadraturePoints();

        auto attr = get_param<std::string>("attr");
        if(!verts.hasProperty(attr)){
            fmt::print("the input verts have no specified channel : {}\n",attr);
            throw std::runtime_error("the input verts have no specified channel");
        }

        // auto weight = get_param<std::string>("weight");
        // if(!verts.hasProperty(weight)){
        //     fmt::print("the input vertices have no specified weight channel : {}\n",weight);
        //     throw std::runtime_error("the input vertices have no specified weight channel");
        // }

        int simplex_size = quads.getChannelSize("inds");
        int attr_dim = verts.getChannelSize(attr);

        quads.append_channels(cudaPol,{{attr,attr_dim}});

        cudaPol(range(quads.size()),
            [verts = proxy<space>({},verts),quads = proxy<space>({},quads),attr = SmallString(attr),simplex_size,attr_dim]
                __device__(int ei) mutable {
                    for(int i = 0;i != attr_dim;++i)
                        quads(attr,i,ei) = 0.0;

                    for(int i  = 0;i != simplex_size;++i){
                        auto idx = reinterpret_bits<int>(quads("inds",i,ei));
                        for(int j = 0;j != attr_dim;++j){
                            quads(attr,j,ei) += verts(attr,j,idx);
                        }
                    }
        });

        set_output("field",field);
    }
};

ZENDEFNODE(ZSSampleVertAttr2Quadrature,{
    {"field"},
    {"field"},
    {
        {"string","attr","attr"}
    },
    {"ZSGeometry"}
});


struct ZSNormalizeVectorField : zeno::INode {
    using tiles_t = typename ZenoParticles::particles_t;

    void apply() override {
        using namespace zs;
        auto field = get_input<ZenoParticles>("field");
        auto type = get_param<std::string>("type");
        auto attr = get_param<std::string>("attr");

        tiles_t& data = (type == "vertex") ? field->getParticles() : field->getQuadraturePoints();

        auto cudaPol = cuda_exec();
        constexpr auto space = execspace_e::cuda;

        int attr_dim = data.getChannelSize(attr);

        cudaPol(range(data.size()),
            [data = proxy<space>({},data),attr_dim,attr = SmallString(attr)] __device__(int di) mutable {
                float length = 0;
                for(int i = 0;i != attr_dim;++i){
                    length += data(attr,i,di) * data(attr,i,di);
                }
                length = zs::sqrt(length) + 1e-6;
                for(int i = 0;i != attr_dim;++i)
                    data(attr,i,di) /= length;
            });

        set_output("field",field);
    }
};

ZENDEFNODE(ZSNormalizeVectorField,{
    {"field"},
    {"field"},
    {
        {"enum vertex quad","type","vertex"},{"string","attr","attr"}
    },
    {"ZSGeometry"}
});

// a temporary node for sampling a quadrature field to another quadrature using bvh
// which should  replaced by a more versatile neighbor-wrangle
struct ZSGaussianNeighborQuadatureSampler : zeno::INode {
    using vec3 = zs::vec<float,3>;
    using bv_t = zs::AABBBox<3, float>;

    constexpr float gauss_kernel(float dist,float sigma) {
        // using namespace zs;
        auto distds = dist/sigma;
        return 1/(sigma /** zs::sqrt(2*zs::g_pi)*/) * zs::exp(-0.5 * distds * distds);
    }

    template <typename TileVecT, int codim = 3>
    zs::Vector<zs::AABBBox<3, typename TileVecT::value_type>>
    retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TileVecT &vtemp,
                            const zs::SmallString &xTag,
                            const typename ZenoParticles::particles_t &eles,
                            zs::wrapv<codim>, int voffset) {
        using namespace zs;
        using T = typename TileVecT::value_type;
        using bv_t = AABBBox<3, T>;
        static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
        constexpr auto space = execspace_e::cuda;
        zs::Vector<bv_t> ret{eles.get_allocator(), eles.size()};
        pol(range(eles.size()), [eles = proxy<space>({}, eles),
                                bvs = proxy<space>(ret),
                                vtemp = proxy<space>({}, vtemp),
                                codim_v = wrapv<codim>{}, xTag,
                                voffset] ZS_LAMBDA(int ei) mutable {
            constexpr int dim = RM_CVREF_T(codim_v)::value;
            auto inds =
                eles.template pack<dim>("inds", ei).template reinterpret_bits<int>() +
                voffset;
            auto x0 = vtemp.template pack<3>(xTag, inds[0]);
            bv_t bv{x0, x0};
            for (int d = 1; d != dim; ++d)
            merge(bv, vtemp.template pack<3>(xTag, inds[d]));
            bvs[ei] = bv;
        });
        return ret;
    }

    template<typename Pol,typename VertBuffer,typename QuadBuffer,typename CenterBuffer>
    void evaluate_quadrature_centers(Pol& pol,VertBuffer& verts,QuadBuffer& quads,CenterBuffer& centers,const std::string& xtag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;   

        // fmt::print("begin evaluate centers\n");

        pol(range(centers.size()),
            [src_centers = proxy<space>(centers),
                    src_quads = proxy<space>({},quads),
                    src_verts = proxy<space>({},verts),
                    xtag = SmallString(xtag)] __device__(int ei) mutable {
                int simplex_size = src_quads.propertySize("inds");
                src_centers[ei] = vec3::zeros();
                // if(ei == 1)
                //     printf("simplex_size : %d\n",simplex_size);
                for(int i = 0;i != simplex_size;++i){
                    auto idx = reinterpret_bits<int>(src_quads("inds",i,ei));
                    // if(ei == 1)
                    //     printf("idx : %d\n",idx);
                    src_centers[ei] += src_verts.pack<3>(xtag,idx) / simplex_size;
                }
        });       

        // fmt::print("finish evaluate centers\n");  
    }

    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto src_field = get_input<ZenoParticles>("source");
        auto dst_field = get_input<ZenoParticles>("dest");

        int use_append = get_input2<int>("use_append");

        auto& src_quads = src_field->getQuadraturePoints();
        auto& src_verts = src_field->getParticles();
        auto& dst_quads = dst_field->getQuadraturePoints();
        auto& dst_verts = dst_field->getParticles();

        auto radius_shrink = get_input2<float>("radius");
        auto mark = get_input2<float>("mark");
        auto mark_tag = get_param<std::string>("mark_tag");

        // auto bvh_thickness = get_param<float>("bvh_thickness");

        int simplex_size = src_quads.getChannelSize("inds");
        if(simplex_size != 4){
            fmt::print("currently, only the tetrahedron is supported\n");
            throw std::runtime_error("currently, only the tetrahedron is supported");
        }

        auto xtag = get_param<std::string>("xtag");
        auto attr = get_param<std::string>("attr");
        auto sigma = get_param<float>("sigma");

        if(!src_quads.hasProperty(attr)){
            fmt::print("the input source quadrature does not have specified channel {}\n",attr);
            throw std::runtime_error("the input source quadrature does not have specified channel");
        }
        if(!dst_quads.hasProperty(attr)){
            fmt::print("the input dest quadrature does not have specified channel {}\n",attr);
            throw std::runtime_error("the input dest quadrature does not have specified channel");
        }
        if(!dst_quads.hasProperty(mark_tag)){
            fmt::print("the input dest quadrature does not have the specified mark tag {}\n",mark_tag);
            throw std::runtime_error("the input dest quadrature does not have the specified mark tag");
        }

        int attr_dim = src_quads.getChannelSize(attr);
        // dst_quads.append_channels(cudaPol,{{attr,attr_dim}});


        // initialize the buffer data to 0
        // cudaPol(range(dst_quads.size()),
        //     [dst_quads = proxy<space>({},dst_quads),attr = SmallString(attr),attr_dim] ZS_LAMBDA(int ei) mutable {
        //             for(int i = 0;i != attr_dim;++i)
        //                 dst_quads(attr,i,ei) = 0.0;
        // }); 

        // fmt::print("initial dst field\n");
        // build bvh for source
        auto bvs = retrieve_bounding_volumes(cudaPol,src_verts,xtag,src_quads,wrapv<4>{},0);
        auto quadsBvh = LBvh<3,32,int,float>{};
        quadsBvh.build(cudaPol,bvs);

        // fmt::print("finish setup bvh\n");
        // initial two buffers containing the quads' centers
        zs::Vector<vec3> src_centers(src_quads.get_allocator(),src_quads.size());
        zs::Vector<vec3> dst_centers(dst_quads.get_allocator(),dst_quads.size());

        // fmt::print("size check {} {} {}\n",src_quads.size(),src_verts.size(),src_centers.size());
        // fmt::print("size check {} {} {}\n",dst_quads.size(),dst_verts.size(),dst_centers.size());
        evaluate_quadrature_centers(cudaPol,src_verts,src_quads,src_centers,xtag);
        evaluate_quadrature_centers(cudaPol,dst_verts,dst_quads,dst_centers,xtag);

        // auto sigma2 = sigma*sigma;

        // fmt::print("finish evaluating the quadrature centers\n");

        cudaPol(range(dst_quads.size()),
            [ dst_quads = proxy<space>({},dst_quads),src_quads = proxy<space>({},src_quads),
                dst_verts = proxy<space>({},dst_verts),src_verts = proxy<space>({},src_verts),
                src_centers = proxy<space>(src_centers),dst_centers = proxy<space>(dst_centers),
                attr = SmallString(attr),xtag = SmallString(xtag),simplex_size,attr_dim,
                bvh = proxy<space>(quadsBvh),sigma,this,use_append,radius_shrink,mark_tag = SmallString(mark_tag),mark] __device__(int di) mutable {
                    if(!use_append)
                        for(int i = 0;i != attr_dim;++i)
                            dst_quads(attr,i,di) = 0.0;
                    else{
                        float field_norm = 0.f;
                        for(int i = 0;i != attr_dim;++i)
                            field_norm += dst_quads(attr,i,di) * dst_quads(attr,i,di);
                        field_norm = zs::sqrt(field_norm);
                        if(field_norm > 1e-6)
                            return;
                    }
                    // compute the center of the src tet
                    auto dst_ct = dst_centers[di]; 
                    float radius = 0;

                    float w_sum = 0;

                    // automatically detected the approapiate radius size
                    for(int i = 0; i != simplex_size;++i){
                        auto idx = reinterpret_bits<int>(dst_quads("inds",i,di));
                        auto dst_vert = dst_verts.pack<3>(xtag,idx);
                        auto dc_dist = (dst_vert - dst_ct).norm();
                        radius = radius < dc_dist ? dc_dist : radius;
                    }

                    radius *= radius_shrink;
                    // float alpha = zs::sqrt((float)zs::g_pi);
                    // if(di == 0){
                    //     printf("check %f %f\n",(float)alpha,(float)zs::g_pi);
                    // }

                    auto dst_bv = bv_t{get_bounding_box(dst_ct - radius, dst_ct + radius)};
                    bvh.iter_neighbors(dst_bv,[&](int si){
                        auto src_ct = src_centers[si];
                        auto dist = (src_ct - dst_ct).norm();
                        auto w = gauss_kernel(dist,sigma);
                        // float distds = dist/sigma;


                        // float beta = zs::exp(-0.5 * distds * distds);
                        // w = 1/(sigma /* zs::sqrt(2*zs::g_pi)*/) * zs::exp(-0.5 * distds * distds);

                        w_sum += w;
                        // printf("sample neighbor : %d->%d %f %f %f\n",si,di,(float)w,(float)alpha,(float)zs::g_pi);
                        for(int i = 0;i != attr_dim;++i)
                            dst_quads(attr,i,di) += w * src_quads(attr,i,si);

                        dst_quads(mark_tag,di) = mark;
                    });

                    // if(w_sum < 1e-6){
                    //     printf("lost element %d\n",di);
                    // }
                    for(int i = 0;i != attr_dim;++i)
                        dst_quads(attr,i,di) /= (w_sum + 1e-6);
        });




        set_output("dest",dst_field);
    }
};

ZENDEFNODE(ZSGaussianNeighborQuadatureSampler,{
    {"source","dest",{"int","use_append","0"},{"float","radius","1"},{"float","mark","-1.0"}},
    {"dest"},
    {
        {"string","mark_tag","mark_tag"},
        {"string","attr","attr"},
        {"string","xtag","x"},
        {"float","sigma","1"}
    },
    {"ZSGeometry"}
});

struct ZSAppendAttribute : zeno::INode {
    using tiles_t = typename ZenoParticles::particles_t;

    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        auto particles = get_input<ZenoParticles>("ZSParticles");
        auto attr = get_param<std::string>("attr");
        auto attr_dim = get_param<int>("attr_dim");

        auto type = get_param<std::string>("type");

        auto fill_val = get_input2<float>("fill");

        tiles_t& data = (type == "particle") ? particles->getParticles() : particles->getQuadraturePoints();

        data.append_channels(cudaPol,{{attr,attr_dim}});
        
        cudaPol(range(data.size()),
            [data = proxy<space>({},data),attr = SmallString(attr),attr_dim,fill_val] __device__(int vi) mutable {
                for(int i = 0;i != attr_dim;++i)
                    data(attr,i,vi) = fill_val;
        });

        set_output("ZSParticles",particles);
    }
};

ZENDEFNODE(ZSAppendAttribute,{
    {"ZSParticles",{"float","fill","0.0"}},
    {"ZSParticles"},
    {
        {"string","attr","attr"},
        {"int","attr_dim","1"},
        {"enum particle quadature","type","particle"}
    },
    {"ZSGeometry"}
});


};