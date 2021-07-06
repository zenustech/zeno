#include "Render.h"

Render::Render() {}

Render::~Render() {}

void Render::preprocess()
{
    std::cout << "*******************PreProcess***********************" << std::endl;
    Mesh3D newframe(m_vtk_path, 0);
    clock_t start, end;
    start = clock();
    newframe.DualMeshTest();
    end = clock();
    std::cout << "========DualMesh Done! total time elapsed:"
              << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;

    // make a copy for debug purpose..
    vertices_copy.resize(newframe.num_of_vertices + newframe.num_of_tets * 4 + newframe.num_of_faces * 3 + newframe.num_of_edges * 2);
    for (size_t k = 0; k < vertices_copy.size(); k++)
    {
        vertices_copy[k] = newframe.vertices[k];
    }
    std::cout << "total pts num: " << vertices_copy.size() << std::endl;

    start = clock();
    CrackSmoothTest(newframe, max_smooth_iter_bound, max_smooth_iter_int);
    end = clock();
    std::cout << "=========Crack Smooth Done! total time elapsed:"
              << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;

    // make a copy for debug purpose..
    vertices_initial.resize(
        newframe.num_of_vertices + newframe.num_of_tets * 4 + newframe.num_of_faces * 3 + newframe.num_of_edges * 2);
    for (size_t k = 0; k < vertices_initial.size(); k++)
    {
        vertices_initial[k] = newframe.vertices[k];
    }

    origmesh = newframe;

    WriteFile(newframe, "uncutmesh", 0);

    std::cout << "done!" << std::endl;
}

void Render::process(int frame)
{
    std::cout << "************Rendering Frame " << frame << "************" << std::endl;

    clock_t start, end;

    start = clock();

    std::cout << "Load File" << std::endl;
    LoadFile(frame);
    std::cout << "Load File done" << std::endl;
    Mesh3D &newframe = origmesh;
    newframe.UpdateVertices(old_points);

    end = clock();
    std::cout << "Update Done! runtime:"
              << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;

    start = clock();

    newframe.TopoEvolution(edge_strech_threshold);
    newframe.UpdateFaceBreak();

    end = clock();
    std::cout << "Topology Update runtime:"
              << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;

    start = clock();

    UpdateCoreF(newframe, frame);

    end = clock();
    std::cout << "F runtime:" << (double)(end - start) / (double)CLOCKS_PER_SEC
              << "s" << std::endl;

    start = clock();

    Sewing(newframe);

    end = clock();
    std::cout << "core vertices sewing runtime:"
              << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    
    start = clock();
    WriteFile(newframe, "mesh", frame);
    end = clock();
    std::cout << "Write cut mesh done! runtime:"
              << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
}

void Render::UpdateCoreF(Mesh3D &newmesh, int frame)
{
    for (int i = 0; i < newmesh.num_of_vertices; i++)
    {
        Eigen::Matrix3d U, V;
        Eigen::Matrix3d Z = new_Fs[i];

        int count = 0;
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        for (size_t k = 0; k < newmesh.vert_link[i].size(); k++)
        {
            auto pe = newmesh.vert_link[i][k];
            int q = (pe->x == i ? pe->y : pe->x);
            if (pe->is_break == false)
            {
                A += (newmesh.vertices[q] - newmesh.vertices[i]) * (vertices_initial[q] - vertices_initial[i]).transpose();
                count++;
            }
        }
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (count >= 2)
        {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd2(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Vector3d sigma = svd2.singularValues();
            U = svd2.matrixU();
            V = svd2.matrixV();
            if (U.determinant() < 0)
            {
                U.col(2) *= -1;
                sigma(2) *= -1;
            }
            if (V.determinant() < 0)
            {
                V.col(2) *= -1;
                sigma(2) *= -1;
            }
            R = U * V.transpose();
        }
        else
        {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Z, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Vector3d sigma = svd.singularValues();
            U = svd.matrixU();
            V = svd.matrixV();
            if (U.determinant() < 0)
            {
                U.col(2) *= -1;
                sigma(2) *= -1;
            }
            if (V.determinant() < 0)
            {
                V.col(2) *= -1;
                sigma(2) *= -1;
            }
            R = U * V.transpose();
        }
        for (size_t j = 0; j < newmesh.core_single[i].size(); j++)
        {
            newmesh.vertices[newmesh.core_single[i][j]] = R * (vertices_initial[newmesh.core_single[i][j]] - vertices_initial[i]) + newmesh.vertices[i];
        }
    }
}

void Render::Sewing(Mesh3D &newmesh)
{
    // take average
    clock_t start, end;

    std::vector<Eigen::Vector3d> &vert = newmesh.vertices;

    start = clock();

    for (size_t i = 0; i < newmesh.EdgetList.size(); i++)
    {
        auto pe = newmesh.EdgetList[i];
        if (pe->is_break == 0)
        {
            int p = pe->core_id(0), q = pe->core_id(1);
            newmesh.vertices[p] = newmesh.vertices[q] = (newmesh.vertices[p] + newmesh.vertices[q]) / 2;
        }
    }
    end = clock();
    std::cout << "Sewing Edge center done! runtime:"
              << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    start = clock();
    for (size_t i = 0; i < newmesh.FacetList.size(); i++)
    {
        auto pf = newmesh.FacetList[i];

        int a = pf->is_break(0), b = pf->is_break(1), c = pf->is_break(2),
            aa = pf->core_id(0), bb = pf->core_id(1), cc = pf->core_id(2);

        if ((a == 0 && b == 0) || (a == 0 && c == 0) || (b == 0 && c == 0))
        {
            vert[aa] = vert[bb] = vert[cc] = (vert[aa] + vert[bb] + vert[cc]) / 3;
        }
        else if (a == 0)
        {
            vert[bb] = vert[cc] = (vert[bb] + vert[cc]) / 2;
        }
        else if (b == 0)
        {
            vert[aa] = vert[cc] = (vert[aa] + vert[cc]) / 2;
        }
        else if (c == 0)
        {
            vert[aa] = vert[bb] = (vert[aa] + vert[bb]) / 2;
        }
    }
    end = clock();
    std::cout << "Sewing Face center done! runtime:"
              << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    start = clock();
    int nv = newmesh.num_of_vertices;
    for (int i = 0; i < newmesh.num_of_tets; i++)
    {
        int s_e = newmesh.num_of_vertices + 4 * newmesh.num_of_tets + 3 * newmesh.num_of_faces;
        Eigen::Matrix4i &e_m = newmesh.tet_2_edge[i];
        int id_a = (e_m(0, 1) - s_e) / 2, id_b = (e_m(0, 2) - s_e) / 2,
            id_c = (e_m(0, 3) - s_e) / 2, id_d = (e_m(1, 2) - s_e) / 2,
            id_e = (e_m(1, 3) - s_e) / 2, id_f = (e_m(2, 3) - s_e) / 2;
        int a = (newmesh.EdgetList[id_a]->is_break == 1 ? 0 : 1),
            b = (newmesh.EdgetList[id_b]->is_break == 1 ? 0 : 1),
            c = (newmesh.EdgetList[id_c]->is_break == 1 ? 0 : 1),
            d = (newmesh.EdgetList[id_d]->is_break == 1 ? 0 : 1),
            e = (newmesh.EdgetList[id_e]->is_break == 1 ? 0 : 1),
            f = (newmesh.EdgetList[id_f]->is_break == 1 ? 0 : 1);

        if (a + b + c + d + e + f >= 4)
        {
            vert[nv + 4 * i] = vert[nv + 4 * i + 1] = vert[nv + 4 * i + 2] = vert[nv + 4 * i + 3] = (vert[nv + 4 * i] + vert[nv + 4 * i + 1] + vert[nv + 4 * i + 2] + vert[nv + 4 * i + 3]) / 4;
        }
        else if (a + b + c + d + e + f == 3 && a + b + d < 3 && a + c + e < 3 && b + c + f < 3 && d + e + f < 3)
        {
            vert[nv + 4 * i] = vert[nv + 4 * i + 1] = vert[nv + 4 * i + 2] = vert[nv + 4 * i + 3] = (vert[nv + 4 * i] + vert[nv + 4 * i + 1] + vert[nv + 4 * i + 2] + vert[nv + 4 * i + 3]) / 4;
        }
        else if (a + b + d == 3 || a + b == 2 || a + d == 2 || b + d == 2) // x-y-z
        {
            vert[nv + 4 * i] = vert[nv + 4 * i + 1] = vert[nv + 4 * i + 2] = (vert[nv + 4 * i] + vert[nv + 4 * i + 1] + vert[nv + 4 * i + 2]) / 3;
        }
        else if (a + c + e == 3 || a + c == 2 || a + e == 2 || c + e == 2) // x-y-w
        {
            vert[nv + 4 * i] = vert[nv + 4 * i + 1] = vert[nv + 4 * i + 3] = (vert[nv + 4 * i] + vert[nv + 4 * i + 1] + vert[nv + 4 * i + 3]) / 3;
        }
        else if (b + c + f == 3 || b + c == 2 || b + f == 2 || c + f == 2) // x-z-w
        {
            vert[nv + 4 * i] = vert[nv + 4 * i + 2] = vert[nv + 4 * i + 3] = (vert[nv + 4 * i] + vert[nv + 4 * i + 2] + vert[nv + 4 * i + 3]) / 3;
        }
        else if (d + e + f == 3 || d + e == 2 || d + f == 2 || e + f == 2) // y-z-w
        {
            vert[nv + 4 * i + 3] = vert[nv + 4 * i + 1] = vert[nv + 4 * i + 2] = (vert[nv + 4 * i + 3] + vert[nv + 4 * i + 1] + vert[nv + 4 * i + 2]) / 3;
        }
        else
        {
            if (a == 1) // x-y
            {
                vert[nv + 4 * i] = vert[nv + 4 * i + 1] = (vert[nv + 4 * i] + vert[nv + 4 * i + 1]) / 2;
            }
            if (b == 1) // x-z
            {
                vert[nv + 4 * i] = vert[nv + 4 * i + 2] = (vert[nv + 4 * i] + vert[nv + 4 * i + 2]) / 2;
            }
            if (c == 1) // x-w
            {
                vert[nv + 4 * i] = vert[nv + 4 * i + 3] = (vert[nv + 4 * i] + vert[nv + 4 * i + 3]) / 2;
            }
            if (d == 1) // y-z
            {
                vert[nv + 4 * i + 1] = vert[nv + 4 * i + 2] = (vert[nv + 4 * i + 1] + vert[nv + 4 * i + 2]) / 2;
            }
            if (e == 1) // y-w
            {
                vert[nv + 4 * i + 1] = vert[nv + 4 * i + 3] = (vert[nv + 4 * i + 1] + vert[nv + 4 * i + 3]) / 2;
            }
            if (f == 1) // z-w
            {
                vert[nv + 4 * i + 2] = vert[nv + 4 * i + 3] = (vert[nv + 4 * i + 2] + vert[nv + 4 * i + 3]) / 2;
            }
        }
    }
    end = clock();
    std::cout << "Sewing Tet center done! runtime:"
              << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
}

std::vector<int> FindCore(std::vector<int> &core, int core_id)
{

    std::vector<int> r;
    for (size_t i = 0; i < core.size(); i++)
    {
        if (core[i] == core_id)
        {
            if (i % 3 == 0)
            {
                r.push_back(core[i + 1]);
                r.push_back(core[i + 2]);
            }
            else if (i % 3 == 1)
            {
                r.push_back(core[i - 1]);
                r.push_back(core[i + 1]);
            }
            else
            {
                r.push_back(core[i - 2]);
                r.push_back(core[i - 1]);
            }
        }
    }

    std::vector<int> out = ArrangeCore(r);
    return out;
}

Eigen::Vector3d intersect(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c, Eigen::Vector3d d)
{
    double ad = (a - d).norm(), cd = (c - d).norm(), ac = (a - c).norm(),
           bd = (b - d).norm(), bc = (b - c).norm();
    double alpha = acos((pow(ad, 2) + pow(cd, 2) - pow(ac, 2)) / (2 * ad * cd));
    double beta = acos((pow(bd, 2) + pow(cd, 2) - pow(bc, 2)) / (2 * bd * cd));

    double a1 = -ad * sin(alpha), a2 = ad * cos(alpha), b1 = bd * sin(beta),
           b2 = bd * cos(beta);

    double m = a2 - (a2 - b2) / (a1 - b1) * a1;

    return d + (c - d) * m / cd;
}

void Render::CrackSmoothTest(Mesh3D &origmesh, int max_bound, int max_int)
{
    clock_t start, end;

    Mesh3D newmesh(origmesh);

    start = clock();
    std::vector<double> edge_max_strech;
    edge_max_strech.resize(newmesh.num_of_edges, 0);
    for (int i = start_frame; i <= end_frame; i++)
    {
        LoadFile(i);
        std::cout << i << std::endl;
        for (int j = 0; j < newmesh.num_of_edges; j++)
        {
            auto pe = newmesh.EdgetList[j];
            int x = pe->x, y = pe->y;
            double temp = (old_points[x] - old_points[y]).norm();
            if (temp > edge_max_strech[j])
                edge_max_strech[j] = temp;
        }
    }
    for (int j = 0; j < newmesh.num_of_edges; j++)
    {
        auto pe = newmesh.EdgetList[j];
        if (edge_max_strech[j] / pe->orig_dist > edge_strech_threshold)
        {
            pe->is_break = 1;
            newmesh.BreakCore(j);
        }
    }
    newmesh.UpdateFaceBreak();
    end = clock();
    std::cout << "run all frame done! runtime:"
              << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;

    start = clock();
    ProcessOutput(newmesh);
    end = clock();
    std::cout << "Process for output done! runtime:"
              << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;
    std::vector<Facet *> &f = newmesh.FacetList;
    std::vector<Eigen::Vector3d> &vert = origmesh.vertices;

    int s_f = newmesh.s_f;
    int s_e = newmesh.s_e;
    int nv = newmesh.num_of_vertices;

    /****************crack smooth Boundary curve*********************/
    start = clock();

    std::vector<Eigen::Vector2i> link; //(ecore,fcore);
    std::vector<std::vector<int>> e_center;
    std::vector<Eigen::Vector3i> f_center;
    for (size_t i = 0; i < newmesh.FacetList.size(); i++)
    {
        auto pf = newmesh.FacetList[i];
        if (pf->is_surf == true)
        {
            int a = f[i]->x;
            int b = f[i]->y;
            int c = f[i]->z;

            int fa = pf->core_id(0), fb = pf->core_id(1), fc = pf->core_id(2);
            int aa = pf->is_break(0), bb = pf->is_break(1), cc = pf->is_break(2);
            if (aa == 1 && bb == 1 && cc == 0)
            {
                int i0 = fc, i1 = origmesh.FindEdge(c, a), i2 = origmesh.FindEdge(c, b);
                f_center.push_back(Eigen::Vector3i(fa, i1, i2));
                link.push_back(Eigen::Vector2i((i1 - s_e) / 2, (i0 - s_f) / 3));
                link.push_back(Eigen::Vector2i((i2 - s_e) / 2, (i0 - s_f) / 3));
            }
            else if (aa == 0 && bb == 1 && cc == 1)
            {
                int i0 = fa, i1 = origmesh.FindEdge(a, b), i2 = origmesh.FindEdge(a, c);
                f_center.push_back(Eigen::Vector3i(fa, i1, i2));
                link.push_back(Eigen::Vector2i((i1 - s_e) / 2, (i0 - s_f) / 3));
                link.push_back(Eigen::Vector2i((i2 - s_e) / 2, (i0 - s_f) / 3));
            }
            else if (aa == 1 && bb == 0 && cc == 1)
            {
                int i0 = fb, i1 = origmesh.FindEdge(b, a), i2 = origmesh.FindEdge(b, c);
                f_center.push_back(Eigen::Vector3i(fa, i1, i2));
                link.push_back(Eigen::Vector2i((i1 - s_e) / 2, (i0 - s_f) / 3));
                link.push_back(Eigen::Vector2i((i2 - s_e) / 2, (i0 - s_f) / 3));
            }
        }
    }

    for (size_t i = 0; i < newmesh.FacetList.size(); i++)
    {
        auto pf = newmesh.FacetList[i];
        if (pf->is_surf == true)
        {
            int x = pf->x, y = pf->y, z = pf->z;

            if (newmesh.FindEdgePtr(x, y)->is_break == 1)
            {
                int e_id = (origmesh.FindEdge(x, y) - s_e) / 2;
                std::vector<int> f_id;
                for (size_t k = 0; k < link.size(); k++)
                {
                    if (link[k][0] == e_id)
                        f_id.push_back(link[k][1]);
                }
                if (f_id.size() == 2)
                {
                    std::vector<int> temp = {s_e + 2 * e_id,
                                             s_e + 2 * e_id + 1,
                                             s_f + 3 * f_id[0],
                                             s_f + 3 * f_id[1],
                                             x,
                                             y};
                    e_center.push_back(temp);
                }
            }
            if (newmesh.FindEdgePtr(x, z)->is_break == 1)
            {
                int e_id = (origmesh.FindEdge(x, z) - s_e) / 2;
                std::vector<int> f_id;
                for (size_t k = 0; k < link.size(); k++)
                {
                    if (link[k][0] == e_id)
                        f_id.push_back(link[k][1]);
                }
                if (f_id.size() == 2)
                {
                    std::vector<int> temp = {s_e + 2 * e_id,
                                             s_e + 2 * e_id + 1,
                                             s_f + 3 * f_id[0],
                                             s_f + 3 * f_id[1],
                                             x,
                                             z};
                    e_center.push_back(temp);
                }
            }
            if (newmesh.FindEdgePtr(y, z)->is_break == 1)
            {
                int e_id = (origmesh.FindEdge(y, z) - s_e) / 2;
                std::vector<int> f_id;
                for (size_t k = 0; k < link.size(); k++)
                {
                    if (link[k][0] == e_id)
                        f_id.push_back(link[k][1]);
                }
                if (f_id.size() == 2)
                {
                    std::vector<int> temp = {s_e + 2 * e_id,
                                             s_e + 2 * e_id + 1,
                                             s_f + 3 * f_id[0],
                                             s_f + 3 * f_id[1],
                                             y,
                                             z};
                    e_center.push_back(temp);
                }
            }
        }
    }

    for (int time = 0; time < max_bound; time++)
    {
        for (size_t i = 0; i < f_center.size(); i++)
        {
            int core_id = f_center[i][0];
            Eigen::Vector3d temp = (vert[f_center[i][1]] + vert[f_center[i][2]]) / 2;
            temp = (temp - vert[core_id]) * smooth_size + vert[core_id];
            vert[core_id] = vert[core_id + 1] = vert[core_id + 2] = temp;
        }
        for (size_t i = 0; i < e_center.size(); i++)
        {
            Eigen::Vector3d temp = intersect(vert[e_center[i][2]], vert[e_center[i][3]],
                                             vert[e_center[i][4]], vert[e_center[i][5]]);
            temp = (temp - vert[e_center[i][0]]) * smooth_size + vert[e_center[i][0]];
            vert[e_center[i][0]] = vert[e_center[i][1]] = temp;
        }
    }

    end = clock();
    std::cout << "Smooth Boundary Done! runtime:"
              << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;

    /****************crack smooth interior surface********************/
    std::vector<Eigen::Vector4i> cut_list;

    for (size_t i = 0; i < newmesh.EdgetList.size(); i++)
    {
        auto pe = newmesh.EdgetList[i];
        if (pe->is_break == 1)
        {
            int core_id = pe->core_id(0);

            for (size_t k = 0; k < newmesh.core_quad[pe->x].size(); k += 4)
            {
                if (newmesh.core_quad[pe->x][k] == core_id)
                {
                    int f1 = newmesh.core_quad[pe->x][k + 1],
                        t = newmesh.core_quad[pe->x][k + 2],
                        f2 = newmesh.core_quad[pe->x][k + 3];

                    core_id = (core_id - s_e) / 2 * 2 + s_e;
                    f1 = (f1 - s_f) / 3 * 3 + s_f;
                    f2 = (f2 - s_f) / 3 * 3 + s_f;
                    t = (t - nv) / 4 * 4 + nv;
                    cut_list.push_back(Eigen::Vector4i(core_id, f1, t, f2));
                }
            }
        }
    }

    std::cout << "Constructed cutlist: " << cut_list.size() << std::endl;

    std::set<int> bound_list;
    for (size_t i = 0; i < newmesh.FacetList.size(); i++)
    {
        auto pf = newmesh.FacetList[i];
        if (pf->is_surf == true)
        {
            bound_list.insert(pf->core_id(0));
        }
    }
    for (size_t i = 0; i < newmesh.EdgetList.size(); i++)
    {
        auto pe = newmesh.EdgetList[i];
        if (pe->is_surfedge == true)
        {
            bound_list.insert(pe->core_id(0));
        }
    }

    std::unordered_map<int, std::vector<int>> vertex_neighbors;
    size_t max_neighbor_count = 0;
    for (Eigen::Vector4i &element : cut_list)
    {
        for (int p = 0; p < 4; ++p)
        {
            int incident_vertex = element(p);
            if (bound_list.find(incident_vertex) != bound_list.end())
                continue;
            auto iter = vertex_neighbors.find(incident_vertex);
            if (iter == vertex_neighbors.end())
            {
                vertex_neighbors[incident_vertex] = std::vector<int>();
                iter = vertex_neighbors.find(incident_vertex);
            }

            for (int q = 0; q < 4; ++q)
            {
                if (q == p)
                    continue;
                if (std::find(iter->second.begin(), iter->second.end(), element(q)) == iter->second.end())
                    iter->second.emplace_back(element(q));
            }
            if (iter->second.size() > max_neighbor_count)
                max_neighbor_count = iter->second.size();
        }
    }

    std::cout << "SurfaceSmoothener: Constructed vertex_neighbors with "
              << vertex_neighbors.size() << " entries. Vertices have at most "
              << max_neighbor_count << " neighbors. " << std::endl;

    start = clock();
    double smooth_scale = 0;
    for (int smooth_iter = 0; smooth_iter < max_int; ++smooth_iter)
    {
        for (auto it = vertex_neighbors.cbegin(); it != vertex_neighbors.cend();
             ++it)
        {
            int incident_vertex = it->first;
            int count = 0;
            Eigen::Vector3d average = Eigen::Vector3d::Zero();
            for (int neighbor_vertex : it->second)
            {
                average += origmesh.vertices[neighbor_vertex];
                count++;
            }
            average *= (double)1 / (double)count;
            Eigen::Vector3d original_value = origmesh.vertices[incident_vertex];
            origmesh.vertices[incident_vertex] = smooth_scale * original_value + ((double)1 - smooth_scale) * average;
        }
    }
    for (auto it = vertex_neighbors.cbegin(); it != vertex_neighbors.cend();
         ++it)
    {
        int incident_vertex = it->first;
        if (incident_vertex < s_f)
        {
            origmesh.vertices[incident_vertex + 1] = origmesh.vertices[incident_vertex + 2] = origmesh.vertices[incident_vertex + 3] = origmesh.vertices[incident_vertex];
        }
        else if (incident_vertex < s_e)
        {
            origmesh.vertices[incident_vertex + 1] = origmesh.vertices[incident_vertex + 2] = origmesh.vertices[incident_vertex];
        }
        else
        {
            origmesh.vertices[incident_vertex + 1] = origmesh.vertices[incident_vertex];
        }
    }

    end = clock();
    std::cout << "Smooth Interior Done! runtime:"
              << (double)(end - start) / (double)CLOCKS_PER_SEC << "s" << std::endl;

    //=========================recover changed
    // data=================================//
    for (auto fp : origmesh.FacetList)
    {
        fp->is_break = Eigen::Vector3i(0, 0, 0);
    }

    for (auto fe : origmesh.EdgetList)
    {
        fe->is_break = 0;
        fe->max_strech = 0;
    }
}

void Render::LoadFile(int frame /*= 0*/)
{
    if (use_binary == true)
    {
        // Warning: NO EMPTY LINE IN TXT FILE!!!!
        std::ifstream infile;
        std::string str;
        std::stringstream ss;
        ss << frame;
        ss >> str;
        str = inputpath  + str + ".dat";
        // std::cout<<str<<std::endl;
        int num;
        infile.open(str, std::ios::binary | std::ios::in);
        // infile >> num;
        infile.read((char *)&num, sizeof(int));
        // std::cout<<" "<<num<<std::endl;
        int size = 0;

        old_points.clear();
        new_Fs.clear();
        for (int i = 0; i < num; i++)
        {
            Eigen::Vector3f p;
            Eigen::Matrix3f temp;
            infile.read((char *)&p, sizeof(float) * 3);
            infile.read((char *)&temp, sizeof(float) * 9);
            Eigen::Matrix3d tempd;
            tempd << temp(0, 0), temp(0, 1), temp(0, 2), temp(1, 0), temp(1, 1),
                temp(1, 2), temp(2, 0), temp(2, 1), temp(2, 2);

            if (pid_min >= 0 && pid_max >= 0 && !(i <= pid_max && i >= pid_min))
            {
                // skip
            }
            else
            {
                old_points.push_back(Eigen::Vector3d(p(0), p(1), p(2)));
                new_Fs.push_back(tempd);
                size++;
            }
        }
        infile.close();
    }
    else
    {
        // Warning: NO EMPTY LINE IN TXT FILE!!!!
        std::ifstream infile;
        std::string str;
        std::stringstream ss;
        ss << frame;
        ss >> str;
        str = inputpath + fileprefix + str + ".txt";
        int num;
        infile.open(str);
        infile >> num;
        int size = 0;
        old_points.clear();
        new_Fs.clear();
        while (!infile.eof())
        {
            double px, py, pz;
            Eigen::Matrix3d temp;
            infile >> px >> py >> pz;
            // std::cout << px << "  " << py << std::endl;
            infile >> temp(0, 0) >> temp(0, 1) >> temp(0, 2) >> temp(1, 0) >> temp(1, 1) >> temp(1, 2) >> temp(2, 0) >> temp(2, 1) >> temp(2, 2);
            old_points.push_back(Eigen::Vector3d(px, py, pz));
            new_Fs.push_back(temp);
            size++;
        }
        infile.close();
    }
}

void Render::WriteFile(Mesh3D &newmesh, std::string filename, int frame /*= 0*/)
{
    Mesh3D *m = &newmesh;

    output_quad.clear();
    output_quad = output_mesh_backup;
    // boundary surf quad..

    std::string buffer = outputpath + filename;
    std::ofstream outfile, fout;
    std::string over = ".obj";
    std::stringstream ss;
    std::string str;
    ss << frame;
    ss >> str;
    str += over;
    str = buffer + str;
    outfile.open(str);
    std::vector<Eigen::Vector3i> tri;
    std::vector<Eigen::Vector4i> quad;
    tri.clear();

    for (int i = 0; i < newmesh.num_of_vertices; i++)
    {
        for (int jj : newmesh.crck_surf_quad[i])
        {
            int j = jj * 4;
            int v1 = m->core_quad[i][j], v2 = m->core_quad[i][j + 1],
                v3 = m->core_quad[i][j + 2], v4 = m->core_quad[i][j + 3];

            Eigen::Vector3d p1 = vertices_copy[v1], p2 = vertices_copy[v2],
                            p3 = vertices_copy[v3], p0 = vertices_copy[i];

            Eigen::Matrix4d A;
            A.col(0) = Eigen::Vector4d(1, p0(0), p0(1), p0(2));
            A.col(1) = Eigen::Vector4d(1, p1(0), p1(1), p1(2));
            A.col(2) = Eigen::Vector4d(1, p2(0), p2(1), p2(2));
            A.col(3) = Eigen::Vector4d(1, p3(0), p3(1), p3(2));
            if (A.determinant() < 0)
            {
                int v = v1;
                v1 = v3;
                v3 = v;
            }
            output_quad.push_back(Eigen::Vector4i(v1, v2, v3, v4));
        }
    }

    int nv = newmesh.num_of_vertices;
    int s_f = newmesh.s_f;
    int s_e = newmesh.s_e;
    std::vector<Eigen::Vector3d> &vert = m->vertices;
    double eps = 1e-6;
    for (int i = 0; i < (int)output_quad.size(); i++)
    {
        for (int j = 0; j < 4; j++)
        {
            int &core_id = output_quad[i](j);
            if (core_id < nv)
            {
                continue;
            }
            else if (core_id < s_f)
            {
                int ss = (core_id - nv) / 4 * 4 + nv;
                if ((vert[ss] - vert[core_id]).norm() < eps)
                {
                    core_id = ss;
                }
                else if ((vert[ss + 1] - vert[core_id]).norm() < eps)
                {

                    core_id = ss + 1;
                }
                else if ((vert[ss + 2] - vert[core_id]).norm() < eps)
                {
                    core_id = ss + 2;
                }
            }
            else if (core_id < s_e)
            {
                int ss = (core_id - s_f) / 3 * 3 + s_f;
                if ((vert[ss] - vert[core_id]).norm() < eps)
                {
                    core_id = ss;
                }
                else if ((vert[ss + 1] - vert[core_id]).norm() < eps)
                {
                    core_id = ss + 1;
                }
            }
            else
            {
                int ss = (core_id - s_e) / 2 * 2 + s_e;
                if ((vert[ss] - vert[core_id]).norm() < eps)
                {
                    core_id = ss;
                }
            }
        }
    }
    vertices.resize(0);
    for (size_t i = 0; i < vertices_copy.size(); i++)
    {
        if (output_vert_flag[i] > 0)
        {
            vertices.push_back(Eigen::Vector3d(m->vertices[i].x(),m->vertices[i].y(),m->vertices[i].z()));
            outfile << "v " << m->vertices[i].x() << " " << m->vertices[i].y() << " "
                    << m->vertices[i].z() << std::endl;
        }
    }
    outfile << std::endl;
    if (clean_debris > 0)
    {
        // std::unordered_map<int, int> vertex_to_component;
        // std::vector<std::vector<int>> component_to_vertex;
        // int num_of_component, num_of_debris = 0;
        // num_of_component = AnalyzeConnectedComponent(output_quad, vertex_to_component, component_to_vertex);
        // for (size_t k = 0; k < component_to_vertex.size(); k++)
        // {
        //     if ((int)component_to_vertex[k].size() < clean_debris)
        //         num_of_debris++;
        // }
        // std::cout << "Get " << num_of_component << " Components, out of which " << num_of_debris << " are debris." << std::endl;
    }
    tris.resize(0);
    for (size_t i = 0; i < output_quad.size(); i++)
    {
        if (output_quad[i][0] == -1)
            continue;
        int idx0 = output_vert_flag[output_quad[i].x()] - 1;
        int idx1 = output_vert_flag[output_quad[i].y()] - 1;
        int idx2 = output_vert_flag[output_quad[i].z()] - 1;
        int idx3 = output_vert_flag[output_quad[i].w()] - 1;
        tris.push_back(Eigen::Vector3i(idx0, idx1, idx2));
        tris.push_back(Eigen::Vector3i(idx2, idx3, idx0));
        outfile << "f " << output_vert_flag[output_quad[i].x()] << " "
                << output_vert_flag[output_quad[i].y()] << " "
                << output_vert_flag[output_quad[i].z()] << " "
                << output_vert_flag[output_quad[i].w()] << std::endl;
    }
    outfile.close();
}

void Render::ProcessOutput(Mesh3D &newmesh)
// Preprocess for output data...
{
    int nv = newmesh.num_of_vertices;
    output_vert_flag.resize(vertices_copy.size(), 0);

    std::vector<Eigen::Vector3d> vert;

    std::vector<std::vector<int>> merg;

    // loop for all edge center and determine draw or not...
    for (size_t i = 0; i < newmesh.EdgetList.size(); i++)
    {
        auto pe = newmesh.EdgetList[i];
        if (pe->is_surfedge == true)
        {
            if (pe->is_break)
            {
                output_vert_flag[pe->core_id(0)] = output_vert_flag[pe->core_id(1)] = 1;
            }
            else
                output_vert_flag[pe->core_id(0)] = 1;
        }
        else // not surf edge...
        {
            if (pe->is_break)
            {
                output_vert_flag[pe->core_id(0)] = output_vert_flag[pe->core_id(1)] = 1;
            }
        }
    }

    // loop for all face center and determine draw or not...
    for (size_t i = 0; i < newmesh.FacetList.size(); i++)
    {
        auto pf = newmesh.FacetList[i];
        int f0 = pf->is_break(0), f1 = pf->is_break(1), f2 = pf->is_break(2);
        int fa = pf->core_id(0), fb = pf->core_id(1), fc = pf->core_id(2);

        if (pf->is_surf == true)
        {
            if (f0 + f1 + f2 <= 1)
            {
                output_vert_flag[fa] = 1;
            }
            else if (f0 == 0)
            {
                // a  b-c
                output_vert_flag[fa] = 1;
                output_vert_flag[fb] = 1;
            }
            else if (f1 == 0)
            {
                // a-c b
                output_vert_flag[fa] = 1;
                output_vert_flag[fb] = 1;
            }
            else if (f2 == 0)
            {
                // a-b c
                output_vert_flag[fa] = 1;
                output_vert_flag[fc] = 1;
            }
            else
            {
                output_vert_flag[fa] = 1;
                output_vert_flag[fb] = 1;
                output_vert_flag[fc] = 1;
            }
        }
        else // not a surface...
        {
            if (f0 + f1 + f2 == 0)
            {
                continue;
            }
            if (f0 + f1 + f2 == 1)
            {
                output_vert_flag[fa] = 1;
                merg.push_back({fa, fb, fc});
            }
            else if (f0 == 0)
            {
                output_vert_flag[fa] = 1, output_vert_flag[fb] = 1;
                merg.push_back({fb, fc});
            }
            else if (f1 == 0)
            {
                output_vert_flag[fa] = 1, output_vert_flag[fb] = 1;
                merg.push_back({fa, fc});
            }
            else if (f2 == 0)
            {
                output_vert_flag[fa] = 1, output_vert_flag[fc] = 1;
                merg.push_back({fa, fb});
            }
            else
            {
                output_vert_flag[fa] = 1, output_vert_flag[fb] = 1,
                output_vert_flag[fc] = 1;
            }
        }
    }

    // loop for all tet center and determine draw or not...
    for (int i = 0; i < newmesh.num_of_tets; i++)
    {
        int s_e = newmesh.num_of_vertices + 4 * newmesh.num_of_tets + 3 * newmesh.num_of_faces;
        Eigen::Matrix4i &e_m = newmesh.tet_2_edge[i];
        int id_a = (e_m(0, 1) - s_e) / 2, id_b = (e_m(0, 2) - s_e) / 2,
            id_c = (e_m(0, 3) - s_e) / 2, id_d = (e_m(1, 2) - s_e) / 2,
            id_e = (e_m(1, 3) - s_e) / 2, id_f = (e_m(2, 3) - s_e) / 2;
        int a = (newmesh.EdgetList[id_a]->is_break == 1 ? 0 : 1),
            b = (newmesh.EdgetList[id_b]->is_break == 1 ? 0 : 1),
            c = (newmesh.EdgetList[id_c]->is_break == 1 ? 0 : 1),
            d = (newmesh.EdgetList[id_d]->is_break == 1 ? 0 : 1),
            e = (newmesh.EdgetList[id_e]->is_break == 1 ? 0 : 1),
            f = (newmesh.EdgetList[id_f]->is_break == 1 ? 0 : 1);

        if (a + b + c + d + e + f == 6)
        {
            // std::cout << "aaaaaaaaa" << std::endl;
            continue;
        }

        int ta = nv + 4 * i, tb = nv + 4 * i + 1, tc = nv + 4 * i + 2,
            td = nv + 4 * i + 3;
        if (a + b + c + d + e + f >= 4)
        {
            output_vert_flag[ta] = 1;
            merg.push_back({ta, tb, tc, td});
        }
        else if (a + b + c + d + e + f == 3 && a + b + d < 3 && a + c + e < 3 && b + c + f < 3 && d + e + f < 3)
        {
            output_vert_flag[ta] = 1;
            merg.push_back({ta, tb, tc, td});
        }
        else if (a + b + d == 3 || a + b == 2 || a + d == 2 || b + d == 2) // x-y-z
        {
            output_vert_flag[ta] = 1;
            output_vert_flag[td] = 1;
            merg.push_back({ta, tb, tc});
        }
        else if (a + c + e == 3 || a + c == 2 || a + e == 2 || c + e == 2) // x-y-w
        {
            output_vert_flag[ta] = 1;
            output_vert_flag[tc] = 1;
            merg.push_back({ta, tb, td});
        }
        else if (b + c + f == 3 || b + c == 2 || b + f == 2 || c + f == 2) // x-z-w
        {
            output_vert_flag[ta] = 1;
            output_vert_flag[tb] = 1;
            merg.push_back({ta, tc, td});
        }
        else if (d + e + f == 3 || d + e == 2 || d + f == 2 || e + f == 2) // y-z-w
        {
            output_vert_flag[tb] = 1;
            output_vert_flag[ta] = 1;
            merg.push_back({tb, tc, td});
        }
        else
        {
            if (a == 1) // x-y
            {
                merg.push_back({ta, tb});
                ta = tb = nv + 4 * i;
            }
            if (b == 1) // x-z
            {
                merg.push_back({ta, tc});
                ta = tc = nv + 4 * i;
            }
            if (c == 1) // x-w
            {
                merg.push_back({ta, td});
                ta = td = nv + 4 * i;
            }
            if (d == 1) // y-z
            {
                merg.push_back({tb, tc});
                tb = tc = nv + 4 * i + 1;
            }
            if (e == 1) // y-w
            {
                merg.push_back({tb, td});
                tb = td = nv + 4 * i + 1;
            }
            if (f == 1) // z-w
            {
                merg.push_back({tc, td});
                tc = td = nv + 4 * i + 2;
            }
            output_vert_flag[ta] = output_vert_flag[tb] = output_vert_flag[tc] = output_vert_flag[td] = 1;
        }
    }

    // boundary sim particle----always need to draw!!!!
    for (size_t i = 0; i < newmesh.FacetList.size(); i++)
    {
        auto pf = newmesh.FacetList[i];
        if (pf->is_surf == true)
        {
            int a = pf->x, b = pf->y, c = pf->z;
            output_vert_flag[a] = output_vert_flag[b] = output_vert_flag[c] = 1;
        }
    }

    // store all the surface quad need to draw
    for (int i = 0; i < newmesh.num_of_tets; i++)
    {
        int x = newmesh.tets[i].x(), y = newmesh.tets[i].y(),
            z = newmesh.tets[i].z(), w = newmesh.tets[i].w();

        Eigen::Matrix4i &f_m = newmesh.tet_2_face[i];
        int xyz = (newmesh.FacetList[f_m(3, 3)]->core_id(0) - newmesh.s_f) / 3;
        output_vert_flag[x] = output_vert_flag[y] = output_vert_flag[z] = 1;
        std::vector<Eigen::Vector4i> temp = (newmesh.WriteFaceHelperquadTest(vertices_copy, xyz, w));
        for (size_t j = 0; j < temp.size(); j++)
            output_quad.push_back(temp[j]);

        int yzw = (newmesh.FacetList[f_m(0, 0)]->core_id(0) - newmesh.s_f) / 3;
        temp = (newmesh.WriteFaceHelperquadTest(vertices_copy, yzw, x));
        for (size_t j = 0; j < temp.size(); j++)
            output_quad.push_back(temp[j]);

        int zwx = (newmesh.FacetList[f_m(1, 1)]->core_id(0) - newmesh.s_f) / 3;
        temp = (newmesh.WriteFaceHelperquadTest(vertices_copy, zwx, y));
        for (size_t j = 0; j < temp.size(); j++)
            output_quad.push_back(temp[j]);

        int wxy = (newmesh.FacetList[f_m(2, 2)]->core_id(0) - newmesh.s_f) / 3;
        temp = (newmesh.WriteFaceHelperquadTest(vertices_copy, wxy, z));
        for (size_t j = 0; j < temp.size(); j++)
            output_quad.push_back(temp[j]);
    }

    // creat a list for output indexing
    int s = 1;
    for (size_t i = 0; i < output_vert_flag.size(); i++)
    {
        if (output_vert_flag[i] == 1)
        {
            output_vert_flag[i] = s;
            s++;
        }
    }

    num_of_surf_quad = output_quad.size();
    output_mesh_backup = output_quad;
}

void Render::ParametersIn(std::string cmd_input_path)
// Load Parameters
{
    std::ifstream infile;
    infile.open("config.txt");
    if (!infile.is_open())
    {
        std::cout << "Could Not Open File!!!";
        exit(EXIT_FAILURE);
    }

    std::string st;
    infile >> st;
    while (infile.good())
    {
        if (st.compare("smooth_iter") == 0)
        {
            infile >> max_smooth_iter_int;
            max_smooth_iter_bound = max_smooth_iter_int / 3;
            if (max_smooth_iter_bound > 20)
            {
                max_smooth_iter_bound = 20;
            }
        }

        if (st.compare("edge_strech") == 0)
        {
            infile >> edge_strech_threshold;
        }

        if (st.compare("path") == 0)
        {
            infile >> inputpath;
            if (cmd_input_path != "")
                inputpath = cmd_input_path;
            outputpath = inputpath;
        }

        if (st.compare("start_frame") == 0)
        {
            infile >> start_frame;
        }

        if (st.compare("end_frame") == 0)
        {
            infile >> end_frame;
        }

        if (st.compare("fileprefix") == 0)
        {
            infile >> fileprefix;
        }

        if (st.compare("clean_debris") == 0)
        {
            infile >> clean_debris;
        }

        if (st.compare("use_binary") == 0)
        {
            infile >> use_binary;
        }

        if (st.compare("pid") == 0)
        {
            infile >> pid_min >> pid_max;
            std::cout << "PID: [" << pid_min << " - " << pid_max << "]" << std::endl;
        }

        infile >> st;
    }
    if (infile.eof())
        std::cout << "Parameters Loaded!!" << std::endl;
    infile.close();
}