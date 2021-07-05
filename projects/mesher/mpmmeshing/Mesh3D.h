#pragma once

#include "help.h"

class Facet
{
public:
    Facet(int x, int y, int z, int x_id, int y_id, int z_id)
        : x(x), y(y), z(z), is_surf(true), flag(false), core_id(Eigen::Vector3i(x_id, y_id, z_id)), is_break(0, 0, 0){};
    int x, y, z;
    bool is_surf;
    bool flag;

    Eigen::Vector3i core_id;
    Eigen::Vector3i is_break; // initial as 0 0 0

    // some added data structure...
    // Edget* yz, zx, zy;
    Eigen::Vector3i edge_id; // edge_index of y-z/z-x/x-y
};

class Edget
{
public:
    Edget(int x, int y)
        : x(x), y(y), max_strech(0), orig_dist(0), is_break(0), is_surfedge(false){};
    Edget(int x, int y, double dist, int x_id, int y_id)
        : x(x), y(y), max_strech(0), orig_dist(dist), is_break(0), core_id(Eigen::Vector2i(x_id, y_id)), is_surfedge(false){};

    int x, y;

    double max_strech;
    double orig_dist;

    bool is_break; // initial as 0

    Eigen::Vector2i core_id;

    bool is_surfedge;
};

class Mesh3D
{
public:
    Mesh3D();
    Mesh3D(std::string strv, std::string strt); // read vert&tet directly from file
    Mesh3D(std::vector<Eigen::Vector3d> &vert, std::vector<Eigen::Vector4i> &tet);
    Mesh3D(std::string filename, int filetype);
    ~Mesh3D();

    int num_of_vertices;
    int num_of_tets;

    int num_of_faces;
    int num_of_edges;

    int s_t, s_f, s_e,
        num_total; // start index of tet\face\edge center respectively

    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector4i> tets;

    // using hash...
    std::unordered_map<std::pair<int, int>, int, hash_pair> edge_map;
    std::unordered_map<key_f, int, key_hash, key_equal> face_map;

    std::vector<Facet *> FacetList;
    std::vector<Edget *> EdgetList;

    std::vector<int> intface, surface;
    std::vector<int> intvec, survec;

    std::vector<std::vector<int>> crck_surf; // crack surface stored at each particle..//id
        // in core triple...
    std::vector<std::vector<int>> crck_surf_quad;
    std::vector<Eigen::Vector3i> crck_bound;

    std::vector<Eigen::Matrix3d> R, F;

    bool is_survec(int vec);

    std::vector<double> distance_original;
    std::vector<double> max_strech;

    std::vector<std::vector<int>> core_quad; // store core index in order of quad at each
        // sim particle     like:e,f,t,f'.....

    std::vector<std::set<int>> core_set; // don't use anymore..
    std::vector<std::vector<int>>
        core_single; // store core index without repetition at each sim particle
    // std::vector<std::unordered_set<int>> core_set;

    int FindEdge(int p, int q);
    int FindEdgeId(int p, int q);
    Edget *FindEdgePtr(int p, int q);
    int FindFace(int x, int y, int z);
    // std::vector<int> FindFaceId(int x, int y);//find face like (u,x,y) or
    // (u,y,x)....
    Facet *FindFacePtr(int x, int y, int z); // find face like(x,y,z) or (x,z,y)

    void PrintParticle();
    void PrintTet();
    void PrintEdge();
    void PrintFace();
    void PrintCore();

    void Test();
    void DualMeshTest();

    void PreProcess();

    // update vertices position
    void UpdateVertices(std::vector<Eigen::Vector3d> &vert);

    void TopoEvolution(double thre = 2.7);
    // void TopoEvolutionwithoutbreak();
    void UpdateFaceBreak();
    void BreakCore(int i, int j); // break vertices i and j
    void BreakCore(int id);       // break id-th edge

    std::vector<std::vector<Edget *>> vert_link;

    std::vector<Eigen::Vector3i> WriteFaceHelper(int f_id, int v_id = -1);
    std::vector<Eigen::Vector4i> WriteFaceHelperquad(int f_id, int v_id = -1);
    std::vector<Eigen::Vector4i> WriteFaceHelperquad(std::vector<Eigen::Vector3d> &vert, int f_id,
                                                     int v_id = -1);
    std::vector<Eigen::Vector4i> WriteFaceHelperquadTest(std::vector<Eigen::Vector3d> &vert, int f_id,
                                                         int v_id = -1);

    void WriteCrckHelper(int &e, int &f, int &t);

    // just for temporary test....
    std::vector<Eigen::Matrix4i> tet_2_face;
    std::vector<Eigen::Matrix4i> tet_2_edge;
    // 0-x 1-y 2-z 3-w   i,j,k,l....
    // tet_2_edge(i,j) equals to FindEdge(i,j) i!=j
    // tet_2_face(i,j) equals to FindFace(j,k,l) or FindFace(j,l,k).. That is find
    // a Facet like (j,k,l) or (j,l,k)
    //
    //
};
