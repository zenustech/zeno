#include <iostream>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <algorithm>
#include <set>
#include <map>
#include <list>
#include <unordered_set>
#include <unordered_map>
#include <typeinfo>
#include <type_traits>
#include <fstream>
#include "MeshIO.hpp"
#include "myPrint.h"
#include "BunnyMeshData.h"
namespace zeno {

BunnyMeshData bunny;


struct ReadVtkTet : INode {
private:
    std::shared_ptr<PrimitiveObject> prim;
    void extractSurf();
public:
  void apply() override {
    auto path = get_input<StringObject>("path")->get();
    prim = std::make_shared<PrimitiveObject>();
    auto &pos = prim->attr<vec3f>("pos");
    auto &quads = prim->quads;

    zs::Mesh<float, 3, int, 4> tet;
    read_tet_mesh_vtk(path, tet);
    const auto numVerts = tet.nodes.size();
    const auto numEles = tet.elems.size();
    prim->resize(numVerts);
    quads.resize(numEles);

    for (int i = 0; i < numVerts; i++)
    {
        pos[i] = tet.nodes[i];
    }
    for (int i = 0; i < numEles; i++)
    {
        quads[i] = tet.elems[i];
    }

    prim->tris.clear();
    extractSurf();
    
    if(has_input<PrimitiveObject>("prim"))
        prim = get_input<PrimitiveObject>("prim");


    set_output("outPrim", std::move(prim));
  }
};

ZENDEFNODE(ReadVtkTet, {/* inputs: */ 
                         {
                            {"readpath", "path"},
                            {"primitive", "prim"}
                         },
                         /* outputs: */
                         {
                             {"primitive", "outPrim"},
                         },
                         /* params: */
                         {},
                         /* category: */
                         {
                             "PBD",
                         }});

   
void ReadVtkTet::extractSurf()
{
    const auto &quads = prim->quads;
    auto &surfs = prim->tris;

    int numTets = quads.size();
    int numFaces = quads.size()*4;
    
    using vec4i = std::array<int,4>;

    //list_faces
    std::vector<vec4i> faces;
    for (int i = 0; i < quads.size(); i++)
    {
        std::array<int,4> tet=quads[i];

        std::sort(tet.begin(),tet.end());
        
        int t0 = tet[0];
        int t1 = tet[1];
        int t2 = tet[2];
        int t3 = tet[3];

        vec4i f0{t0, t1, t2, i};
        vec4i f1{t0, t1, t3, i};
        vec4i f2{t0, t2, t3, i};
        vec4i f3{t1, t2, t3, i};

        faces.push_back(f0);
        faces.push_back(f1);
        faces.push_back(f2);
        faces.push_back(f3);
    }

    auto myLess = [](auto a, auto b){
        return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
    };

    //sort faces
    std::sort(faces.begin(),faces.end(), myLess);


    /* -------------------------------------------------------------------------- */
    /*                       use list to remove shared faces                      */
    /* -------------------------------------------------------------------------- */
    auto myEqual = [](auto a, auto b){
        if((a[0]==b[0])&&(a[1]==b[1])&&(a[2]==b[2])) return true; 
        else return false;};
    //copy to a list
    std::list<vec4i> facelist;
    for(auto it:faces)
        facelist.push_back(it);
    
    //remove shared faces from the list
    auto f_prev = facelist.begin();
    auto f = facelist.begin();  f++;
    for(;f!=facelist.end();)
    {
        if(myEqual(*f, *f_prev))
        {
            f++; //move to the next
            if(f==facelist.end())//if f move to the end(), should break
            {
                facelist.erase(f_prev,f);
                break;
            }
            f_prev = facelist.erase(f_prev,f); //return the next
            f++; // move f to the next of next
        }
        else
        {
            f_prev = f;
            f++;
        }
    }

    //recontruct the surf with orders
    for(auto x:facelist)
    {
        int tetId = x[3];
        std::array<int, 4> vert = quads[tetId];
        std::array<bool, 4> hasVert={false, false, false, false};
        
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                if(x[k] == vert[j])
                    hasVert[j] = true;
            
        if (hasVert[0] &&  hasVert[2] && hasVert[1])
            surfs.push_back(vec3i{vert[0],vert[2],vert[1]}); 

        if (hasVert[0] &&  hasVert[3] && hasVert[2])
            surfs.push_back(vec3i{vert[0],vert[3],vert[2]});

        if (hasVert[0] &&  hasVert[1] && hasVert[3])
            surfs.push_back(vec3i{vert[0],vert[1],vert[3]});

        if (hasVert[1] &&  hasVert[2] && hasVert[3])
            surfs.push_back(vec3i{vert[1],vert[2],vert[3]});
    }

    std::cout<<"before extractSurf numFaces: "<<numFaces<<std::endl;
    std::cout<<"after extractSurf numSurfs: "<<surfs.size()<<std::endl;
}

} // namespace zeno
