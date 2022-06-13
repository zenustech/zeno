#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/logger.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>


void prim_flatten(zeno::PrimitiveObject *prim_in, zeno::PrimitiveObject *prim) {
    prim_in->add_attr<zeno::vec3f>("uv");
    prim->verts.resize(prim_in->tris.size() * 3);
    prim->tris.resize(prim_in->tris.size());
    auto &att_clr = prim->add_attr<zeno::vec3f>("clr");
    auto &att_nrm = prim->add_attr<zeno::vec3f>("nrm");
    auto &att_uv = prim->add_attr<zeno::vec3f>("uv");
    // auto &att_tan = prim->add_attr<zeno::vec3f>("tang");
    bool has_uv = prim_in->tris.has_attr("uv0") && prim_in->tris.has_attr("uv1") && prim_in->tris.has_attr("uv2");

    //std::cout<<"size verts:"<<prim_in->verts.size()<<std::endl;
    auto &in_pos = prim_in->verts;
    auto &in_clr = prim_in->add_attr<zeno::vec3f>("clr");
    auto &in_nrm = prim_in->add_attr<zeno::vec3f>("nrm");
    auto &in_uv = prim_in->attr<zeno::vec3f>("uv");

    for (size_t tid = 0; tid < prim_in->tris.size(); tid++) {
        //std::cout<<tid<<std::endl;
        size_t vid = tid * 3;
        prim->verts[vid] = in_pos[prim_in->tris[tid][0]];
        prim->verts[vid + 1] = in_pos[prim_in->tris[tid][1]];
        prim->verts[vid + 2] = in_pos[prim_in->tris[tid][2]];
        att_clr[vid] = in_clr[prim_in->tris[tid][0]];
        att_clr[vid + 1] = in_clr[prim_in->tris[tid][1]];
        att_clr[vid + 2] = in_clr[prim_in->tris[tid][2]];
        att_nrm[vid] = in_nrm[prim_in->tris[tid][0]];
        att_nrm[vid + 1] = in_nrm[prim_in->tris[tid][1]];
        att_nrm[vid + 2] = in_nrm[prim_in->tris[tid][2]];
        att_uv[vid] = has_uv ? prim_in->tris.attr<zeno::vec3f>("uv0")[tid] : in_uv[prim_in->tris[tid][0]];
        att_uv[vid + 1] = has_uv ? prim_in->tris.attr<zeno::vec3f>("uv1")[tid] : in_uv[prim_in->tris[tid][1]];
        att_uv[vid + 2] = has_uv ? prim_in->tris.attr<zeno::vec3f>("uv2")[tid] : in_uv[prim_in->tris[tid][2]];
        // att_tan[vid]         = prim_in->tris.attr<zeno::vec3f>("tang")[tid];
        // att_tan[vid+1]       = prim_in->tris.attr<zeno::vec3f>("tang")[tid];
        // att_tan[vid+2]       = prim_in->tris.attr<zeno::vec3f>("tang")[tid];
        prim->tris[tid] = zeno::vec3i(vid, vid + 1, vid + 2);
    }
    //flatten here, keep the rest of codes unchanged.
}

namespace zeno{

struct VisualUVObj :  zeno::INode  {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto outprim = new zeno::PrimitiveObject;

        prim_flatten(prim.get(), outprim);
        
        set_output("prim", std::move(std::shared_ptr<zeno::PrimitiveObject>(outprim)));
    }
};

ZENDEFNODE(VisualUVObj,
{   /* inputs: */ 
    {
        "prim",
    }, 
    /* outputs: */ 
    {
        "prim"
    }, 
    /* params: */ 
    {}, 
    /* category: */ 
    {"math",}
});

}
