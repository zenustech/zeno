#include <iostream>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <algorithm>
#include <set>
#include <map>
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
    for (int i = 0; i < numVerts; i++)
    {
        quads[i] = tet.elems[i];
    }

    if(has_input<PrimitiveObject>("prim"))
        prim = get_input<PrimitiveObject>("prim");

    prim->tris.clear();

    echo(prim->tris.size());

    extractSurf();


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
    // /* -------------------------------------------------------------------------- */
    // /*                               list faces and remove shared                 */
    // /* -------------------------------------------------------------------------- */
    std::map<std::set<int>, zeno::vec3i> faces; //key is the sorted face, value is the original face
    std::vector<int> dup_index; //to record the index of tet with duplicated face
    std::vector<zeno::vec3i> dup_face; //to record the original data of duplicated face
    for (int i = 0; i < quads.size(); i++)
    {
        zeno::vec3i f0{quads[i][0], quads[i][2], quads[i][1]};
        zeno::vec3i f1{quads[i][0], quads[i][3], quads[i][2]};
        zeno::vec3i f2{quads[i][0], quads[i][1], quads[i][3]};
        zeno::vec3i f3{quads[i][1], quads[i][2], quads[i][3]};

        std::list<zeno::vec3i> list{f0,f1,f2,f3};
        for(auto& f:list)
        {
            std::set<int> fset(f.begin(), f.end()); //use set to get sorted face, which servers as the key

            auto it = faces.find(fset);
            if(it!=faces.end())  
            {
                faces.erase(it);     //erase the shared faces
                dup_index.push_back(i);     //record the tet index
                dup_face.push_back(it->second); //record the original face data
            }
            else 
            {
                faces[fset] = f;
            }
        }
    }

    for(const auto& f:faces)
        surfs.push_back(f.second);

    std::cout<<"before extractSurf numFaces: "<<numFaces<<std::endl;
    std::cout<<"after extractSurf numSurfs: "<<surfs.size()<<std::endl;
    
    std::ofstream fout;
    fout.open("surf.txt");
    for(const auto& f:surfs)
    {
        for(const auto& x:f)
            fout<<x<<"\t";
        fout<<"\n";
    }
    fout.close();
}

} // namespace zeno