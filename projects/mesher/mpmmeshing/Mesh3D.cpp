#include "Mesh3D.h"

Mesh3D::Mesh3D()
{
    vertices.clear();
    tets.clear();

    // tets = {Eigen::Vector4i()}

    vertices = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
                Eigen::Vector3d(0, 0, 1)};
    num_of_vertices = 4;
    tets = {Eigen::Vector4i(0, 1, 2, 3)};
    num_of_tets = 1;
    crck_surf.resize(num_of_vertices);
    crck_surf_quad.resize(num_of_vertices);
    crck_bound.clear();
}

Mesh3D::Mesh3D(std::string strv, std::string strt)
{
    vertices.clear();
    tets.clear();

    std::ifstream infile;

    infile.open(strv);
    while (!infile.eof())
    {
        double px, py, pz;
        infile >> px >> py >> pz;
        vertices.push_back(Eigen::Vector3d(px, py, pz));
    }
    infile.close();
    num_of_vertices = vertices.size();

    infile.open(strt);
    while (!infile.eof())
    {
        int px, py, pz, pw;
        infile >> px >> py >> pz >> pw;
        tets.push_back(Eigen::Vector4i(px, py, pz, pw));
    }
    infile.close();
    num_of_tets = tets.size();

    crck_surf.resize(num_of_vertices);
    crck_surf_quad.resize(num_of_vertices);
    crck_bound.clear();

    R.resize(num_of_vertices, Eigen::Matrix3d::Identity());
    F.resize(num_of_vertices, Eigen::Matrix3d::Identity());

    std::cout << num_of_vertices << std::endl;
    std::cout << num_of_tets << std::endl;
}

void readTetmeshVtk(std::istream &in, std::vector<Eigen::Vector3d> &X,
                    std::vector<Eigen::Vector4i> &indices)
{
    std::string line;
    Eigen::Vector3d position;
    bool reading_points = false;
    bool reading_tets = false;
    size_t n_points = 0;
    size_t n_tets = 0;
    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        if (line.size() == (size_t)(0))
        {
        }
        else if (line.substr(0, 6) == "POINTS")
        {
            reading_points = true;
            reading_tets = false;
            ss.ignore(128, ' '); // ignore "POINTS"
            ss >> n_points;
        }
        else if (line.substr(0, 5) == "CELLS")
        {
            reading_points = false;
            reading_tets = true;
            ss.ignore(128, ' '); // ignore "CELLS"
            ss >> n_tets;
        }
        else if (line.substr(0, 10) == "CELL_TYPES")
        {
            reading_points = false;
            reading_tets = false;
        }
        else if (reading_points)
        {
            for (size_t i = 0; i < n_points; i++)
            {
                Eigen::Vector3d pos;
                ss >> pos(0) >> pos(1) >> pos(2);
                X.emplace_back(pos);
                std::cout << i << " " << pos.transpose() << std::endl;
            }
        }
        else if (reading_tets)
        {
            for (size_t i = 0; i < n_tets; i++)
            {
                Eigen::Vector4i tet;
                int tmp;
                ss >> tmp;
                ss >> tet(0) >> tet(1) >> tet(2) >> tet(3);
                indices.emplace_back(tet);
            }
        }
    }
}
Mesh3D::Mesh3D(std::string filename, int filetype)
{
    vertices.clear();
    tets.clear();

    switch (filetype)
    {
    case 0:
        // read vtkfile
        {
            std::ifstream fs;
            fs.open(filename);
            //ZIRAN_ASSERT(fs, "could not open ", vtk_file);
            readTetmeshVtk(fs, vertices, tets);
            break;
        }
    case 1:
        // read tet.txt file
        {
            std::ifstream infile;
            int numv, numt;
            infile.open(filename);
            infile >> numv;
            for (int i = 0; i < numv; i++)
            {
                double px, py, pz;
                infile >> px >> py >> pz;
                vertices.push_back(Eigen::Vector3d(px, py, pz));
                // std::cout << px << py << std::endl;
            }
            infile >> numt;
            for (int i = 0; i < numt; i++)
            {
                int px, py, pz, pw;
                infile >> px >> py >> pz >> pw;
                tets.push_back(Eigen::Vector4i(px, py, pz, pw));
                // std::cout<<px<<" "<<py<<" "<<pz<<std::endl;
            }
            infile.close();
            break;
        }
    }

    num_of_vertices = vertices.size();
    num_of_tets = tets.size();
    std::cout << num_of_vertices << std::endl;
    std::cout << num_of_tets << std::endl;
    crck_surf.resize(num_of_vertices);
    crck_surf_quad.resize(num_of_vertices);
    crck_bound.clear();

    R.resize(num_of_vertices, Eigen::Matrix3d::Identity());
    F.resize(num_of_vertices, Eigen::Matrix3d::Identity());
}

Mesh3D::Mesh3D(std::vector<Eigen::Vector3d> &vert, std::vector<Eigen::Vector4i> &tet)
{
    vertices = vert;
    tets = tet;
    num_of_vertices = vert.size();
    num_of_tets = tet.size();
}

Mesh3D::~Mesh3D() {}

bool Mesh3D::is_survec(int vec)
{
    for (size_t i = 0; i < survec.size(); i++)
    {
        if (survec[i] == vec)
        {
            return true;
        }
    }
    return false;
}

int Mesh3D::FindEdge(int p, int q)
{
    Eigen::Vector2i sorted_edge = sort(p, q);
    auto it = edge_map.find(std::make_pair(sorted_edge(0), sorted_edge(1)));
    if (it != edge_map.end())
    {
        auto pe = EdgetList[it->second];
        if (pe->x == p)
        {
            return pe->core_id(0);
        }
        else if (pe->y == p)
        {
            return pe->core_id(1);
        }
        else
        {
            std::cout << "ERROR!!!!" << std::endl;
        }
    }

    return -1;
}

int Mesh3D::FindEdgeId(int p, int q)
{
    Eigen::Vector2i sorted_edge = sort(p, q);
    auto it = edge_map.find(std::make_pair(sorted_edge(0), sorted_edge(1)));
    if (it != edge_map.end())
    {
        return it->second;
    }
    return -1;
}

Edget *Mesh3D::FindEdgePtr(int p, int q)
{
    Eigen::Vector2i sorted_edge = sort(p, q);
    auto it = edge_map.find(std::make_pair(sorted_edge(0), sorted_edge(1)));
    if (it != edge_map.end())
    {
        return EdgetList[it->second];
    }
    return nullptr;
}

int Mesh3D::FindFace(int x, int y, int z)
{
    Eigen::Vector3i sorted_face = sort(x, y, z);
    auto it = face_map.find(std::make_tuple(sorted_face(0), sorted_face(1), sorted_face(2)));
    if (it != face_map.end())
    {
        auto pf = FacetList[it->second];
        if (pf->x == x)
            return pf->core_id(0);
        else if (pf->y == x)
            return pf->core_id(1);
        else if (pf->z == x)
            return pf->core_id(2);
        else
        {
            std::cout << "EROOR!!!" << std::endl;
        }
    }

    return -1;
}

Facet *Mesh3D::FindFacePtr(int x, int y, int z)
{
    Eigen::Vector3i sorted_face = sort(x, y, z);
    auto it = face_map.find(std::make_tuple(sorted_face(0), sorted_face(1), sorted_face(2)));
    if (it != face_map.end())
        return FacetList[it->second];

    return nullptr;
}

void Mesh3D::PrintParticle()
{
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Particle " << i << ": " << vertices[i].x() << " "
                  << vertices[i].y() << " " << vertices[i].z() << std::endl;
    }
}

void Mesh3D::PrintTet()
{
    for (int i = 0; i < num_of_tets; i++)
    {
        std::cout << "Tet " << i << ": " << tets[i].x() << " " << tets[i].y() << " "
                  << tets[i].z() << " " << tets[i].w() << " " << std::endl;
    }
}

void Mesh3D::PrintEdge() {}

void Mesh3D::PrintFace()
{
    for (size_t i = 0; i < FacetList.size(); i++)
    {
        std::cout << "Facet " << i << ": " << FacetList[i]->x << " " << FacetList[i]->y
                  << " " << FacetList[i]->z << " " << FacetList[i]->core_id(0) << " "
                  << FacetList[i]->core_id(1) << " " << FacetList[i]->core_id(2) << " "
                  << std::endl;
    }
}

void Mesh3D::PrintCore()
{
}

void Mesh3D::Test()
{
}

void Mesh3D::DualMeshTest()
{

    clock_t start, end;

    // vertices.resize(5000000);

    // core particle:
    // simulation particles---
    // tet center---
    // face center---
    // edge center---

    int core_id = num_of_vertices;
    //======tet center====//

    start = clock();
    //======face center====//
    // order:yzw--zwx--wxy--xyz

    std::set<int> int_face; // interior face...
    core_id = num_of_vertices + 4 * num_of_tets;
    tet_2_face.resize(num_of_tets, Eigen::Matrix4i::Zero());

    for (int i = 0; i < num_of_tets; i++)
    {
        // if (i % 10 == 0) std::cout << i << std::endl;
        int x = tets[i].x(), y = tets[i].y(), z = tets[i].z(), w = tets[i].w();
        auto temp = FindFacePtr(y, z, w);
        if (temp == nullptr)
        {
            Facet *pf = new Facet(y, z, w, core_id, core_id + 1, core_id + 2);
            FacetList.push_back(pf);
            int faceid = FacetList.size() - 1;
            Eigen::Vector3i sorted_face = sort(y, z, w);
            face_map[std::make_tuple(sorted_face(0), sorted_face(1), sorted_face(2))] = faceid;
            core_id += 3;
            tet_2_face[i](0, 0) = faceid;
        }
        else
        {
            int faceid = (temp->core_id(0) - num_of_vertices - num_of_tets * 4) / 3;
            int_face.insert(faceid);
            temp->is_surf = false;
            tet_2_face[i](0, 0) = faceid;
        }

        temp = FindFacePtr(z, w, x);
        if (temp == nullptr)
        {
            Facet *pf = new Facet(z, w, x, core_id, core_id + 1, core_id + 2);
            FacetList.push_back(pf);
            int faceid = FacetList.size() - 1;
            Eigen::Vector3i sorted_face = sort(z, w, x);
            face_map[std::make_tuple(sorted_face(0), sorted_face(1), sorted_face(2))] = faceid;
            core_id += 3;
            tet_2_face[i](1, 1) = faceid;
        }
        else
        {
            int faceid = (temp->core_id(0) - num_of_vertices - num_of_tets * 4) / 3;
            int_face.insert(faceid);
            temp->is_surf = false;
            tet_2_face[i](1, 1) = faceid;
        }
        temp = FindFacePtr(w, x, y);
        if (temp == nullptr)
        {
            Facet *pf = new Facet(w, x, y, core_id, core_id + 1, core_id + 2);
            FacetList.push_back(pf);
            int faceid = FacetList.size() - 1;
            Eigen::Vector3i sorted_face = sort(w, x, y);
            face_map[std::make_tuple(sorted_face(0), sorted_face(1), sorted_face(2))] = faceid;
            core_id += 3;
            tet_2_face[i](2, 2) = faceid;
        }
        else
        {
            int faceid = (temp->core_id(0) - num_of_vertices - num_of_tets * 4) / 3;
            int_face.insert(faceid);
            temp->is_surf = false;
            tet_2_face[i](2, 2) = faceid;
        }
        temp = FindFacePtr(x, y, z);
        if (temp == nullptr)
        {
            Facet *pf = new Facet(x, y, z, core_id, core_id + 1, core_id + 2);
            FacetList.push_back(pf);
            int faceid = FacetList.size() - 1;
            Eigen::Vector3i sorted_face = sort(x, y, z);
            face_map[std::make_tuple(sorted_face(0), sorted_face(1), sorted_face(2))] = faceid;
            core_id += 3;
            tet_2_face[i](3, 3) = faceid;
        }
        else
        {
            int faceid = (temp->core_id(0) - num_of_vertices - num_of_tets * 4) / 3;
            int_face.insert(faceid);
            temp->is_surf = false;
            tet_2_face[i](3, 3) = faceid;
        }
    }
    num_of_faces = FacetList.size();
    end = clock();
    std::cout << "faces initialized runtime:" << (end - start) / (double)CLOCKS_PER_SEC
              << std::endl;

    start = clock();
    //======edge center======//
    core_id = num_of_vertices + 4 * num_of_tets + 3 * FacetList.size();
    tet_2_edge.resize(num_of_tets, Eigen::Matrix4i::Zero());

    for (int i = 0; i < num_of_tets; i++)
    {
        Eigen::Vector4i &t = tets[i];
        for (int p = 0; p < 4; p++)
        {
            for (int q = p + 1; q < 4; q++)
            {
                auto temp = FindEdgePtr(t(p), t(q));
                if (temp == nullptr)
                {
                    double dist = (vertices[t(p)] - vertices[t(q)]).norm();
                    Edget *pe = new Edget(t(p), t(q), dist, core_id, core_id + 1);
                    EdgetList.push_back(pe);
                    int edgeid = EdgetList.size() - 1;
                    Eigen::Vector2i sorted_edge = sort(t(p), t(q));
                    edge_map[std::make_pair(sorted_edge(0), sorted_edge(1))] = edgeid;
                    tet_2_edge[i](p, q) = core_id, tet_2_edge[i](q, p) = core_id + 1;
                    core_id += 2;
                }
                else
                {
                    if (temp->x == t(p))
                    {
                        tet_2_edge[i](p, q) = temp->core_id(0),
                                         tet_2_edge[i](q, p) = temp->core_id(1);
                    }
                    else if (temp->y == t(p))
                    {
                        tet_2_edge[i](p, q) = temp->core_id(1),
                                         tet_2_edge[i](q, p) = temp->core_id(0);
                    }
                    else
                        std::cout << "ERROR" << std::endl;
                }
            }
        }
    }

    num_of_edges = EdgetList.size();
    end = clock();
    std::cout << "edges initialized runtime:" << (end - start) / (double)CLOCKS_PER_SEC
              << std::endl;

    s_t = num_of_vertices;
    s_f = num_of_vertices + 4 * num_of_tets;
    s_e = num_of_vertices + 4 * num_of_tets + 3 * num_of_faces;
    num_total = num_of_vertices + 4 * num_of_tets + 3 * num_of_faces + 2 * num_of_edges;

    // initialize core vertices position... for tet\face\edge respectively
    vertices.resize(num_total);
    for (int i = 0; i < num_of_tets; i++)
    {
        int x = tets[i].x(), y = tets[i].y(), z = tets[i].z(), w = tets[i].w();
        Eigen::Vector3d center = (vertices[x] + vertices[y] + vertices[z] + vertices[w]) / 4;
        vertices[s_t + i * 4] = vertices[s_t + i * 4 + 1] = vertices[s_t + i * 4 + 2] = vertices[s_t + i * 4 + 3] = center;
    }
    for (int i = 0; i < num_of_faces; i++)
    {
        auto pf = FacetList[i];
        int x = pf->x, y = pf->y, z = pf->z;
        Eigen::Vector3d center = (vertices[x] + vertices[y] + vertices[z]) / 3;
        vertices[s_f + i * 3] = vertices[s_f + i * 3 + 1] = vertices[s_f + i * 3 + 2] = center;
    }
    for (int i = 0; i < num_of_edges; i++)
    {
        auto pe = EdgetList[i];
        int x = pe->x, y = pe->y;
        Eigen::Vector3d center = (vertices[x] + vertices[y]) / 2;
        vertices[s_e + i * 2] = vertices[s_e + i * 2 + 1] = center;
    }

    std::cout << "first part done" << std::endl;

    // lable sim particle as surface or interior
    std::set<int> sur_vec;
    for (auto pf : FacetList)
    {
        if (pf->is_surf == true)
        {
            sur_vec.insert(pf->x);
            sur_vec.insert(pf->y);
            sur_vec.insert(pf->z);

            auto pe = FindEdgePtr(pf->x, pf->y);
            pe->is_surfedge = true;
            pe = FindEdgePtr(pf->x, pf->z);
            pe->is_surfedge = true;
            pe = FindEdgePtr(pf->y, pf->z);
            pe->is_surfedge = true;
        }
    }
    survec.clear();
    intvec.clear();
    for (auto itr = sur_vec.begin(); itr != sur_vec.end(); itr++)
    {
        survec.push_back(*itr);
    }
    for (int j = 0; j < survec[0]; j++)
        intvec.push_back(j);
    for (int i = 0; i < (int)survec.size() - 1; i++)
    {
        for (int j = survec[i] + 1; j < survec[i + 1]; j++)
        {
            intvec.push_back(j);
        }
    }
    for (int j = survec[survec.size() - 1]; j < num_of_vertices; j++)
        intvec.push_back(j);

    start = clock();
    for (int i = 0; i < num_of_tets; i++)
    {
        Eigen::Vector4i &t = tets[i];
        for (int p = 0; p < 4; p++)
        {
            auto pf = FacetList[tet_2_face[i](p, p)];
            for (int q = 0; q < 4; q++)
            {
                if (q == p)
                    continue;
                int temp = 0;
                if (pf->x == t(q))
                    temp = pf->core_id(0);
                else if (pf->y == t(q))
                    temp = pf->core_id(1);
                else if (pf->z == t(q))
                    temp = pf->core_id(2);
                else
                    std::cout << "ERROR" << std::endl;

                tet_2_face[i](p, q) = temp;
            }
        }
    }
    end = clock();
    std::cout << "Assign face and edge to tet. runtime:"
              << (end - start) / (double)CLOCKS_PER_SEC << std::endl;

    // add core index
    // order:yzw--zwx--wxy--xyz
    core_quad.clear();
    core_quad.resize(num_of_vertices);

    start = clock();
    for (int i = 0; i < num_of_tets; i++)
    {
        int x = tets[i].x(), y = tets[i].y(), z = tets[i].z(), w = tets[i].w();
        // quad...
        Eigen::Matrix4i &e_m = tet_2_edge[i];
        Eigen::Matrix4i &f_m = tet_2_face[i];
        /*
        e_m:
            0     1     2     3
        0[      (x,y) (x,z) (x,y)]
        1[(y,x)       (y,z) (y,w)]
        2[(z,x) (z,y)       (z,w)]
        3[(w,x) (w,y) (w,z)      ]

        f_m:
                0		1		2		3
        0[        (y,z,w) (z,y,w) (w,z,y)]
        1[(x,z,w)         (z,x,w) (w,x,z)]
        2[(x,y,w) (y,x,w)         (w,x,y)]
        3[(x,y,z) (y,x,z) (z,x,y)        ]
*/
        // x
        core_quad[x].push_back(e_m(0, 1));
        core_quad[x].push_back(f_m(3, 0));
        core_quad[x].push_back(num_of_vertices + 4 * i);
        core_quad[x].push_back(f_m(2, 0));
        core_quad[x].push_back(e_m(0, 2));
        core_quad[x].push_back(f_m(1, 0));
        core_quad[x].push_back(num_of_vertices + 4 * i);
        core_quad[x].push_back(f_m(3, 0));
        core_quad[x].push_back(e_m(0, 3));
        core_quad[x].push_back(f_m(2, 0));
        core_quad[x].push_back(num_of_vertices + 4 * i);
        core_quad[x].push_back(f_m(1, 0));
        // y
        core_quad[y].push_back(e_m(1, 2));
        core_quad[y].push_back(f_m(3, 1));
        core_quad[y].push_back(num_of_vertices + 4 * i + 1);
        core_quad[y].push_back(f_m(0, 1));
        core_quad[y].push_back(e_m(1, 3));
        core_quad[y].push_back(f_m(2, 1));
        core_quad[y].push_back(num_of_vertices + 4 * i + 1);
        core_quad[y].push_back(f_m(0, 1));
        core_quad[y].push_back(e_m(1, 0));
        core_quad[y].push_back(f_m(3, 1));
        core_quad[y].push_back(num_of_vertices + 4 * i + 1);
        core_quad[y].push_back(f_m(2, 1));
        // z
        core_quad[z].push_back(e_m(2, 3));
        core_quad[z].push_back(f_m(0, 2));
        core_quad[z].push_back(num_of_vertices + 4 * i + 2);
        core_quad[z].push_back(f_m(1, 2));
        core_quad[z].push_back(e_m(2, 0));
        core_quad[z].push_back(f_m(3, 2));
        core_quad[z].push_back(num_of_vertices + 4 * i + 2);
        core_quad[z].push_back(f_m(1, 2));
        core_quad[z].push_back(e_m(2, 1));
        core_quad[z].push_back(f_m(3, 2));
        core_quad[z].push_back(num_of_vertices + 4 * i + 2);
        core_quad[z].push_back(f_m(0, 2));
        // w
        core_quad[w].push_back(e_m(3, 1));
        core_quad[w].push_back(f_m(0, 3));
        core_quad[w].push_back(num_of_vertices + 4 * i + 3);
        core_quad[w].push_back(f_m(2, 3));
        core_quad[w].push_back(e_m(3, 2));
        core_quad[w].push_back(f_m(0, 3));
        core_quad[w].push_back(num_of_vertices + 4 * i + 3);
        core_quad[w].push_back(f_m(1, 3));
        core_quad[w].push_back(e_m(3, 0));
        core_quad[w].push_back(f_m(1, 3));
        core_quad[w].push_back(num_of_vertices + 4 * i + 3);
        core_quad[w].push_back(f_m(2, 3));
    }
    end = clock();
    std::cout << "core initialized! runtime:" << (end - start) / (double)CLOCKS_PER_SEC
              << std::endl;

    start = clock();
    PreProcess();
    end = clock();
    std::cout << "PreProcess done! runtime:" << (end - start) / (double)CLOCKS_PER_SEC
              << std::endl;
    std::cout << "second part done" << std::endl;
}

void Mesh3D::PreProcess()
// PreProcess for future convinience
// to be modified..
{
    clock_t start, end;

    start = clock();
    // add particle neighbors..
    vert_link.resize(num_of_vertices);
    for (auto pe : EdgetList)
    {
        vert_link[pe->x].push_back(pe);
        vert_link[pe->y].push_back(pe);
    }
    end = clock();
    std::cout << "Compute vert neighbor done! runtime:"
              << (end - start) / (double)CLOCKS_PER_SEC << std::endl;

    start = clock();
    core_single.resize(num_of_vertices);
    for (int i = 0; i < num_of_tets; i++)
    {
        int x = tets[i].x(), y = tets[i].y(), z = tets[i].z(), w = tets[i].w();
        core_single[x].push_back(s_t + i * 4);
        core_single[y].push_back(s_t + i * 4 + 1);
        core_single[z].push_back(s_t + i * 4 + 2);
        core_single[w].push_back(s_t + i * 4 + 3);
    }
    for (auto pf : FacetList)
    {
        int x = pf->x, y = pf->y, z = pf->z;
        core_single[x].push_back(pf->core_id(0));
        core_single[y].push_back(pf->core_id(1));
        core_single[z].push_back(pf->core_id(2));
    }
    for (auto pe : EdgetList)
    {
        int x = pe->x, y = pe->y;
        core_single[x].push_back(pe->core_id(0));
        core_single[y].push_back(pe->core_id(1));
    }

    end = clock();
    std::cout << "Creat core set done! runtime:"
              << (end - start) / (double)CLOCKS_PER_SEC << std::endl;

    start = clock();
    // assign each face with edge id..
    for (size_t i = 0; i < FacetList.size(); i++)
    {
        auto pf = FacetList[i];
        int x = pf->x, y = pf->y, z = pf->z;
        pf->edge_id(0) = FindEdgeId(y, z);
        pf->edge_id(1) = FindEdgeId(z, x);
        pf->edge_id(2) = FindEdgeId(x, y);
    }
    end = clock();
    std::cout << "Assign edge to face done! runtime:"
              << (end - start) / (double)CLOCKS_PER_SEC << std::endl;
}

void Mesh3D::UpdateVertices(std::vector<Eigen::Vector3d> &vert)
{
    for (int i = 0; i < num_of_vertices; i++)
    {
        vertices[i] = vert[i];
    }
}

void Mesh3D::TopoEvolution(double thre)
{
    int num = 0;

    for (int i = 0; i < num_of_edges; i++)
    // for (auto pe : EdgetList)
    {
        auto pe = EdgetList[i];
        int x = pe->x, y = pe->y;
        double temp = (vertices[x] - vertices[y]).norm() / pe->orig_dist;

        if (temp > pe->max_strech)
        {
            pe->max_strech = temp;
        }
        if (pe->is_break == 0 && pe->max_strech > thre)
        {
            pe->is_break = 1;
            BreakCore(i);
            num++;
        }
    }
    std::cout << num << std::endl;
}

void Mesh3D::UpdateFaceBreak()
{
    for (size_t i = 0; i < FacetList.size(); i++)
    {
        auto pf = FacetList[i];
        for (int j = 0; j < 3; j++)
        {
            if (EdgetList[pf->edge_id(j)]->is_break == 1)
                pf->is_break(j) = 1;
        }
    }
}

void Mesh3D::BreakCore(int id)
{
    auto pe = EdgetList[id];
    int i = pe->x, j = pe->y;
    int a = pe->core_id(0), b = pe->core_id(1);
    for (size_t k = 0; k < core_quad[i].size() / 4; k++)
    {
        if (core_quad[i][4 * k] == a)
        {
            crck_surf_quad[i].push_back(k);
        }
    }
    for (size_t k = 0; k < core_quad[j].size() / 4; k++)
    {
        if (core_quad[j][4 * k] == b)
        {
            crck_surf_quad[j].push_back(k);
        }
    }
}

void Mesh3D::BreakCore(int i, int j)
{

    int a = FindEdge(i, j), b = FindEdge(j, i);
    for (size_t k = 0; k < core_quad[i].size() / 4; k++)
    {
        if (core_quad[i][4 * k] == a)
        {
            crck_surf_quad[i].push_back(k);
        }
    }
    for (size_t k = 0; k < core_quad[j].size() / 4; k++)
    {
        if (core_quad[j][4 * k] == b)
        {
            crck_surf_quad[j].push_back(k);
        }
    }
}

std::vector<Eigen::Vector3i> Mesh3D::WriteFaceHelper(int f_id, int v_id)
// draw i-th face
{
    std::vector<Eigen::Vector3i> tri;
    auto pf = FacetList[f_id];
    if (pf->is_surf == true)
    {

        int a = pf->x, b = pf->y, c = pf->z, aa = pf->core_id(0),
            bb = pf->core_id(1), cc = pf->core_id(2);
        int f0 = pf->is_break(0), f1 = pf->is_break(1), f2 = pf->is_break(2);

        if (f0 + f1 + f2 > 0)
        {
            int ab = FindEdge(a, b), ba = FindEdge(b, a), ac = FindEdge(a, c),
                ca = FindEdge(c, a), bc = FindEdge(b, c), cb = FindEdge(c, b);

            if (f0 == 0)
                tri.push_back(Eigen::Vector3i(b, c, bb));
            else
            {
                tri.push_back(Eigen::Vector3i(bc, bb, b));
                tri.push_back(Eigen::Vector3i(cb, cc, c));
            }
            if (f1 == 0)
                tri.push_back(Eigen::Vector3i(a, c, aa));
            else
            {
                tri.push_back(Eigen::Vector3i(ac, aa, a));
                tri.push_back(Eigen::Vector3i(ca, cc, c));
            }
            if (f2 == 0)
                tri.push_back(Eigen::Vector3i(a, b, aa));
            else
            {
                tri.push_back(Eigen::Vector3i(ab, aa, a));
                tri.push_back(Eigen::Vector3i(ba, bb, b));
            }
        }
        else
        {
            tri.push_back(Eigen::Vector3i(a, b, c));
        }

        if (v_id != -1)
            for (size_t j = 0; j < tri.size(); j++)
            {
                Eigen::Vector3d p1 = vertices[tri[j][1]], p2 = vertices[tri[j][2]],
                                p3 = vertices[v_id], p0 = vertices[tri[j][0]];
                Eigen::Matrix4d A;
                A.col(0) = Eigen::Vector4d(1, p0(0), p0(1), p0(2));
                A.col(1) = Eigen::Vector4d(1, p1(0), p1(1), p1(2));
                A.col(2) = Eigen::Vector4d(1, p2(0), p2(1), p2(2));
                A.col(3) = Eigen::Vector4d(1, p3(0), p3(1), p3(2));
                // if (p1.cross(p2).dot(p3)>0)
                if (A.determinant() > 0)
                {
                    int v = tri[j][1];
                    tri[j][1] = tri[j][2];
                    tri[j][2] = v;
                }
            }
    }
    return tri;
}

std::vector<Eigen::Vector4i> Mesh3D::WriteFaceHelperquad(int f_id,
                                                         int v_id /*= -1*/)
{
    std::vector<Eigen::Vector4i> quad;
    auto pf = FacetList[f_id];
    if (pf->is_surf == true)
    {
        int a = pf->x, b = pf->y, c = pf->z;
        int fa = pf->core_id(0), fb = pf->core_id(1), fc = pf->core_id(2);
        int ab = FindEdge(a, b), ba = FindEdge(b, a), ac = FindEdge(a, c),
            ca = FindEdge(c, a), bc = FindEdge(b, c), cb = FindEdge(c, b);
        quad.push_back(Eigen::Vector4i(a, ab, fa, ac));
        quad.push_back(Eigen::Vector4i(b, bc, fb, ba));
        quad.push_back(Eigen::Vector4i(c, ca, fc, cb));
        if (v_id != -1)
            for (size_t j = 0; j < quad.size(); j++)
            {
                Eigen::Vector3d p1 = vertices[quad[j][1]], p2 = vertices[quad[j][2]],
                                p3 = vertices[quad[j][3]], p0 = vertices[v_id];
                Eigen::Matrix4d A;
                A.col(0) = Eigen::Vector4d(1, p0(0), p0(1), p0(2));
                A.col(1) = Eigen::Vector4d(1, p1(0), p1(1), p1(2));
                A.col(2) = Eigen::Vector4d(1, p2(0), p2(1), p2(2));
                A.col(3) = Eigen::Vector4d(1, p3(0), p3(1), p3(2));
                if (A.determinant() < 0)
                {
                    int v = quad[j][1];
                    quad[j][1] = quad[j][2];
                    quad[j][2] = v;
                    v = quad[j][0];
                    quad[j][0] = quad[j][3];
                    quad[j][3] = v;
                }
            }
    }
    return quad;
}

std::vector<Eigen::Vector4i> Mesh3D::WriteFaceHelperquad(std::vector<Eigen::Vector3d> &vert,
                                                         int f_id,
                                                         int v_id /*= -1*/)
{
    std::vector<Eigen::Vector4i> quad;
    auto pf = FacetList[f_id];
    if (pf->is_surf == true)
    {
        int a = pf->x, b = pf->y, c = pf->z;
        int fa = pf->core_id(0), fb = pf->core_id(1), fc = pf->core_id(2);
        int ab = FindEdge(a, b), ba = FindEdge(b, a), ac = FindEdge(a, c),
            ca = FindEdge(c, a), bc = FindEdge(b, c), cb = FindEdge(c, b);
        quad.push_back(Eigen::Vector4i(a, ab, fa, ac));
        quad.push_back(Eigen::Vector4i(b, bc, fb, ba));
        quad.push_back(Eigen::Vector4i(c, ca, fc, cb));

        if (v_id != -1)
            for (size_t j = 0; j < quad.size(); j++)
            {
                Eigen::Vector3d p1 = vert[quad[j][1]], p2 = vert[quad[j][2]],
                                p3 = vert[quad[j][3]], p0 = vert[v_id];
                Eigen::Matrix4d A;
                A.col(0) = Eigen::Vector4d(1, p0(0), p0(1), p0(2));
                A.col(1) = Eigen::Vector4d(1, p1(0), p1(1), p1(2));
                A.col(2) = Eigen::Vector4d(1, p2(0), p2(1), p2(2));
                A.col(3) = Eigen::Vector4d(1, p3(0), p3(1), p3(2));
                if (A.determinant() < 0)
                {
                    int v = quad[j][1];
                    quad[j][1] = quad[j][2];
                    quad[j][2] = v;
                    v = quad[j][0];
                    quad[j][0] = quad[j][3];
                    quad[j][3] = v;
                }
            }
    }
    return quad;
}

std::vector<Eigen::Vector4i>
Mesh3D::WriteFaceHelperquadTest(std::vector<Eigen::Vector3d> &vert, int f_id,
                                int v_id /*= -1*/)
{
    // int nv;nv = num_of_vertices;
    // int s_f;s_f = num_of_vertices + num_of_tets * 4;
    int s_e;
    s_e = num_of_vertices + num_of_tets * 4 + num_of_faces * 3;
    std::vector<Eigen::Vector4i> quad;
    auto pf = FacetList[f_id];
    if (pf->is_surf == true)
    {
        int a = pf->x, b = pf->y, c = pf->z;
        int f0 = pf->is_break(0), f1 = pf->is_break(1), f2 = pf->is_break(2);
        int fa = pf->core_id(0), fb = pf->core_id(1), fc = pf->core_id(2);
        int ab = FindEdge(a, b), ba = FindEdge(b, a), ac = FindEdge(a, c),
            ca = FindEdge(c, a), bc = FindEdge(b, c), cb = FindEdge(c, b);
        if (f0 + f1 + f2 <= 1)
        {
            fa = fb = fc = pf->core_id(0);
        }
        else if (f0 == 0)
        {
            fa = pf->core_id(0), fb = fc = pf->core_id(1);
        }
        else if (f1 == 0)
        {
            fa = fc = pf->core_id(0), fb = pf->core_id(1);
        }
        else if (f2 == 0)
        {
            fa = fb = pf->core_id(0), fc = pf->core_id(2);
        }
        else
        {
            fa = pf->core_id(0), fb = pf->core_id(1), fc = pf->core_id(2);
        }

        if (f0 == 0)
        {
            bc = cb = (bc - s_e) / 2 * 2 + s_e;
        }
        if (f1 == 0)
        {
            ac = ca = (ac - s_e) / 2 * 2 + s_e;
        }
        if (f2 == 0)
        {
            ab = ba = (ab - s_e) / 2 * 2 + s_e;
        }

        quad.push_back(Eigen::Vector4i(a, ab, fa, ac));
        quad.push_back(Eigen::Vector4i(b, bc, fb, ba));
        quad.push_back(Eigen::Vector4i(c, ca, fc, cb));

        if (v_id != -1)
            for (size_t j = 0; j < quad.size(); j++)
            {
                Eigen::Vector3d p1 = vert[quad[j][1]], p2 = vert[quad[j][2]],
                                p3 = vert[quad[j][3]], p0 = vert[v_id];
                Eigen::Matrix4d A;
                A.col(0) = Eigen::Vector4d(1, p0(0), p0(1), p0(2));
                A.col(1) = Eigen::Vector4d(1, p1(0), p1(1), p1(2));
                A.col(2) = Eigen::Vector4d(1, p2(0), p2(1), p2(2));
                A.col(3) = Eigen::Vector4d(1, p3(0), p3(1), p3(2));
                // if (p1.cross(p2).dot(p3)>0)
                if (A.determinant() < 0)
                {
                    int v = quad[j][1];
                    quad[j][1] = quad[j][2];
                    quad[j][2] = v;
                    v = quad[j][0];
                    quad[j][0] = quad[j][3];
                    quad[j][3] = v;
                }
            }
    }
    return quad;
}

void Mesh3D::WriteCrckHelper(int &e, int &f, int &t)
{
    int s_t;
    s_t = num_of_vertices;
    int s_f;
    s_f = num_of_vertices + num_of_tets * 4;
    int s_e;
    s_e = num_of_vertices + num_of_tets * 4 + num_of_faces * 3;
    int e_id = (e - s_e) / 2;
    int f_id = (f - s_f) / 3;
    int t_id = (t - s_t) / 4;

    if (EdgetList[e_id]->is_break == 0)
        e = s_e + e_id * 2;

    if ((vertices[s_f + f_id * 3] - vertices[f]).norm() < 1e-6)
    {
        f = s_f + f_id * 3;
    }
    else if ((vertices[s_f + f_id * 3 + 1] - vertices[f]).norm() < 1e-6)
    {
        f = s_f + f_id * 3 + 1;
    }

    if ((vertices[s_t + t_id * 4] - vertices[t]).norm() < 1e-6)
    {
        t = s_t + t_id * 4;
    }
    else if ((vertices[s_t + t_id * 4 + 1] - vertices[t]).norm() < 1e-6)
    {
        t = s_t + t_id * 4 + 1;
    }
    else if ((vertices[s_t + t_id * 4 + 2] - vertices[t]).norm() < 1e-6)
    {
        t = s_t + t_id * 4 + 2;
    }
}
