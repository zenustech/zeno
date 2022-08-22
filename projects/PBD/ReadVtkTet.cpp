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
#include "MeshIO.hpp"

#include "BunnyMeshData.h"

namespace zeno {

template<typename T>
void printPrim(T t, int i);

struct ReadVtkTet : INode {
private:
    BunnyMeshData theBunnyMesh;
    std::shared_ptr<PrimitiveObject> prim;
    // void extractSurf(const zeno::AttrVector<zeno::vec4i> &quads, zeno::AttrVector<zeno::vec3i> &surfs);
    void extractSurf();
public:
  void apply() override {
    auto path = get_input<StringObject>("path")->get();
    prim = std::make_shared<PrimitiveObject>();
    auto &pos = prim->attr<vec3f>("pos");
    auto &quads = prim->quads;
    // auto &surfs = prim->tris;

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

    extractSurf();

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(ReadVtkTet, {/* inputs: */ 
                         {
                            {"readpath", "path"},
                            {"primitive", "prim"}
                         },
                         /* outputs: */
                         {
                             {"primitive", "prim"},
                         },
                         /* params: */
                         {},
                         /* category: */
                         {
                             "PBD",
                         }});


// void ReadVtkTet::extractSurf()
// {
//     auto &quads = prim->quads;
//     auto &surf = prim->tris;

//     int numTets = quads.size();
//     int numFaces = quads.size()*4;
//     /* -------------------------------------------------------------------------- */
//     /*                               list all faces                               */
//     /* -------------------------------------------------------------------------- */
//     std::set<std::set<int>> face;
//     std::vector<int> dup_index;
//     std::vector<std::set<int>> dup_set;
//     for (int i = 0; i < quads.size(); i++)
//     {
//         int a = quads[i][0];
//         int b = quads[i][1];
//         int c = quads[i][2];
//         int d = quads[i][3];

//         std::set f1{a,b,c};
//         std::set f2{a,b,d};
//         std::set f3{a,c,d};
//         std::set f4{b,c,d};

//         for(auto f:{f1,f2,f3,f4})
//         {
//             if(face.count(f)==1) 
//             {
//                 face.erase(f);
//                 dup_index.push_back(i);
//                 dup_set.push_back(f);
//             }
//             else 
//             {
//                 face.insert(f);
//             }
//         }
//     }

//     /* -------------------------------------------------------------------------- */
//     /*                            extract the surfaces                            */
//     /* -------------------------------------------------------------------------- */
//     int numSurfs = face.size();
//     surf.resize(numSurfs);
//     std::cout<<"numFaces: "<<numFaces<<std::endl;
//     std::cout<<"numSurfs: "<<numSurfs<<std::endl;
//     // face.erase(std::unique(face.begin(), face.end()), face.end());
    
//     for (auto &&f : face) {
//         zeno::vec3i tri{};
//         int no = 0;
//         for (auto &&v : f)
//             tri[no++] = v;
//         surf.values.push_back(tri);
//     }
//     printPrim(surf[1],1);
//     printPrim(quads[1],1);
// }


void ReadVtkTet::extractSurf()
{
    auto &quads = prim->quads;
    auto &surf = prim->tris;

    int numTets = quads.size();
    int numFaces = quads.size()*4;
    /* -------------------------------------------------------------------------- */
    /*                               list all faces                               */
    /* -------------------------------------------------------------------------- */
    std::map<std::set<int>, zeno::vec3i> face; //key is the sorted face, value is the original face
    std::vector<int> dup_index; //to record the index of tet with duplicated face
    std::vector<zeno::vec3i> dup_face; //to record the original data of duplicated face
    for (int i = 0; i < quads.size(); i++)
    {
        zeno::vec3i f0{quads[i][0], quads[i][2], quads[i][1]};
        zeno::vec3i f1{quads[i][0], quads[i][3], quads[i][2]};
        zeno::vec3i f2{quads[i][0], quads[i][1], quads[i][3]};
        zeno::vec3i f3{quads[i][1], quads[i][2], quads[i][3]};

        std::list<zeno::vec3i> list{f0,f1,f2,f3};
        for(auto f:list)
        {
            std::set<int> fset(f.begin(), f.end()); //use set to get sorted face number as the key

            auto it = face.find(fset);
            if(it!=face.end())  
            {
                face.erase(it);     //erase the shared faces
                dup_index.push_back(i);     //record the tet index
                dup_face.push_back(it->second); //record the original face data
            }
            else 
            {
                face[fset] = f;
            }
        }
    }

    /* -------------------------------------------------------------------------- */
    /*                            extract the surfaces                            */
    /* -------------------------------------------------------------------------- */
    int numSurfs = face.size();
    surf.resize(numSurfs);
    std::cout<<"numFaces: "<<numFaces<<std::endl;
    std::cout<<"numSurfs: "<<numSurfs<<std::endl;
    
    for(const auto& f:face)
    {
        surf.push_back(f.second);
    }

    printPrim(surf[1],1);
    printPrim(quads[1],1);
}








//helpers
template<typename T>
void printPrim(T t, int i=-1)
{
    if(std::is_same<T, zeno::vec3i>::value)
        std::cout<<"tris["<<i<<"]: ";
    else if (std::is_same<T, zeno::vec4i>::value)
        std::cout<<"quads["<<i<<"]: ";
    
    for(auto x:t)
        std::cout<<x<<"\t";
    std::cout<<"\n";
}


template<typename T>
void printPrim(T t)
{
    int i = 0;
    for(auto x: t)
    {
        printPrim(x, i);
        i++;
    }
}










// struct ExtractMeshSurface : INode {
//   void apply() override {
//     auto prim = get_input<PrimitiveObject>("prim");
//     auto &pos = prim->attr<vec3f>("pos");
//     auto &quads = prim->quads;
//     auto ompExec = zs::omp_exec();
//     const auto numVerts = pos.size();
//     const auto numEles = quads.size();

//     auto op = get_param<std::string>("op");
//     bool includePoints = false;
//     bool includeLines = false;
//     bool includeTris = false;
//     if (op == "all") {
//       includePoints = true;
//       includeLines = true;
//       includeTris = true;
//     } else if (op == "point")
//       includePoints = true;
//     else if (op == "edge")
//       includeLines = true;
//     else if (op == "surface")
//       includeTris = true;

//     std::vector<int> points;
//     std::vector<float> pointAreas;
//     std::vector<vec2i> lines;
//     std::vector<float> lineAreas;
//     std::vector<vec3i> tris;
//     {
//       using namespace zs;
//       zs::HashTable<int, 3, int> surfTable{0};
//       constexpr auto space = zs::execspace_e::openmp;

//       surfTable.resize(ompExec, 4 * numEles);
//       surfTable.reset(ompExec, true);
//       // compute getsurface
//       // std::vector<int> tri2tet(4 * numEles);
//       ompExec(range(numEles), [table = proxy<space>(surfTable),
//                                &quads](int ei) mutable {
//         using table_t = RM_CVREF_T(table);
//         using vec3i = zs::vec<int, 3>;
//         auto record = [&table, ei](const vec3i &triInds) mutable {
//           if (auto sno = table.insert(triInds); sno != table_t::sentinel_v)
//             ; // tri2tet[sno] = ei;
//           else
//             printf("ridiculous, more than one tet share the same surface!");
//         };
//         auto inds = quads[ei];
//         record(vec3i{inds[0], inds[2], inds[1]});
//         record(vec3i{inds[0], inds[3], inds[2]});
//         record(vec3i{inds[0], inds[1], inds[3]});
//         record(vec3i{inds[1], inds[2], inds[3]});
//       });
//       //
//       tris.resize(numEles * 4);
//       Vector<int> surfCnt{1, memsrc_e::host};
//       surfCnt.setVal(0);
//       ompExec(range(surfTable.size()),
//               [table = proxy<space>(surfTable), surfCnt = surfCnt.data(),
//                &tris](int i) mutable {
//                 using vec3i = zs::vec<int, 3>;
//                 auto triInds = table._activeKeys[i];
//                 using table_t = RM_CVREF_T(table);
//                 if (table.query(vec3i{triInds[2], triInds[1], triInds[0]}) ==
//                         table_t::sentinel_v &&
//                     table.query(vec3i{triInds[1], triInds[0], triInds[2]}) ==
//                         table_t::sentinel_v &&
//                     table.query(vec3i{triInds[0], triInds[2], triInds[1]}) ==
//                         table_t::sentinel_v)
//                   tris[atomic_add(exec_omp, surfCnt, 1)] =
//                       zeno::vec3i{triInds[0], triInds[1], triInds[2]};
//               });
//       auto scnt = surfCnt.getVal();
//       tris.resize(scnt);
//       fmt::print("{} surfaces\n", scnt);

//       // surface points
//       HashTable<int, 1, int> vertTable{numVerts};
//       HashTable<int, 2, int> edgeTable{3 * numEles};
//       vertTable.reset(ompExec, true);
//       edgeTable.reset(ompExec, true);
//       ompExec(tris,
//               [vertTable = proxy<space>(vertTable),
//                edgeTable = proxy<space>(edgeTable)](vec3i triInds) mutable {
//                 using vec1i = zs::vec<int, 1>;
//                 using vec2i = zs::vec<int, 2>;
//                 for (int d = 0; d != 3; ++d) {
//                   vertTable.insert(vec1i{triInds[d]});
//                   edgeTable.insert(vec2i{triInds[d], triInds[(d + 1) % 3]});
//                 }
//               });
//       auto svcnt = vertTable.size();
//       points.resize(svcnt);
//       pointAreas.resize(svcnt, 0.f);
//       copy(mem_host, points.data(), vertTable._activeKeys.data(),
//            sizeof(int) * svcnt);
//       fmt::print("{} surface verts\n", svcnt);

//       // surface edges
//       Vector<int> surfEdgeCnt{1};
//       surfEdgeCnt.setVal(0);
//       auto dupEdgeCnt = edgeTable.size();
//       std::vector<int> dupEdgeToSurfEdge(dupEdgeCnt, -1);
//       lines.resize(dupEdgeCnt);
//       ompExec(range(dupEdgeCnt), [edgeTable = proxy<space>(edgeTable), &lines,
//                                   surfEdgeCnt = surfEdgeCnt.data(),
//                                   &dupEdgeToSurfEdge](int edgeNo) mutable {
//         using vec2i = zs::vec<int, 2>;
//         vec2i edge = edgeTable._activeKeys[edgeNo];
//         using table_t = RM_CVREF_T(edgeTable);
//         if (auto eno = edgeTable.query(vec2i{edge[1], edge[0]});
//             eno == table_t::sentinel_v || // opposite edge not exists
//             (eno != table_t::sentinel_v &&
//              edge[0] < edge[1])) { // opposite edge does exist
//           auto no = atomic_add(exec_omp, surfEdgeCnt, 1);
//           lines[no] = zeno::vec2i{edge[0], edge[1]};
//           dupEdgeToSurfEdge[edgeNo] = no;
//         }
//       });
//       auto secnt = surfEdgeCnt.getVal();
//       lines.resize(secnt);
//       lineAreas.resize(secnt, 0.f);
//       fmt::print("{} surface edges\n", secnt);

//       ompExec(tris,
//               [&, vertTable = proxy<space>(vertTable),
//                edgeTable = proxy<space>(edgeTable)](vec3i triInds) mutable {
//                 using vec3 = zs::vec<float, 3>;
//                 using vec1i = zs::vec<int, 1>;
//                 using vec2i = zs::vec<int, 2>;
//                 for (int d = 0; d != 3; ++d) {
//                   auto p0 = vec3::from_array(pos[triInds[0]]);
//                   auto p1 = vec3::from_array(pos[triInds[1]]);
//                   auto p2 = vec3::from_array(pos[triInds[2]]);
//                   float area = (p1 - p0).cross(p2 - p0).norm() / 2;
//                   // surface vert
//                   using vtable_t = RM_CVREF_T(vertTable);
//                   auto vno = vertTable.query(vec1i{triInds[d]});
//                   atomic_add(exec_omp, &pointAreas[vno], area / 3);
//                   // surface edge
//                   using etable_t = RM_CVREF_T(edgeTable);

//           auto eno = edgeTable.query(vec2i{triInds[(d + 1) % 3], triInds[d]});
//           if (auto seNo = dupEdgeToSurfEdge[eno]; seNo != etable_t::sentinel_v)
//             atomic_add(exec_omp, &lineAreas[seNo], area / 3);
//                 }
//               });
//     }
//     if (includeTris)
//       prim->tris.values = tris; // surfaces
//     if (includeLines) {
//       prim->lines.values = lines; // surfaces edges
//       prim->lines.add_attr<float>("area") = lineAreas;
//     }
//     if (includePoints) {
//       prim->points.values = points; // surfaces points
//       prim->points.add_attr<float>("area") = pointAreas;
//     }
//     set_output("prim", std::move(prim));
//   }
// };

// ZENDEFNODE(ExtractMeshSurface, {{{"quad (tet) mesh", "prim"}},
//                                 {{"mesh with surface topos", "prim"}},
//                                 {{"enum all point edge surface", "op", "all"}},
//                                 {"primitive"}});


} // namespace zeno