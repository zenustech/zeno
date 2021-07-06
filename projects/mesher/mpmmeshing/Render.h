#pragma once
#include "Mesh3D.h"
#include "help.h"

class Render
{
public:
    Render();
    ~Render();

    Mesh3D origmesh, oldframe;

    std::vector<Eigen::Vector3d> old_points;
    std::vector<Eigen::Matrix3d> new_Fs;

    void preprocess();

    void process(int frame);

    void UpdateCoreF(Mesh3D &newmesh, int frame);

    void Sewing(Mesh3D &newmesh);

    void CrackSmoothTest(Mesh3D &newmesh, int max_bound = 2, int max_int = 2);

    void LoadFile(int frame = 0); // Load particle position to old_points

    void WriteFile(Mesh3D &newmesh, std::string filename = "", int frame = 0);

    std::string inputpath = "/home/mine/data/1/";
    std::string outputpath = "/home/mine/data/1/";

    int start_frame = 1;
    int end_frame = 44;

    int max_smooth_iter_bound = 50 / 3;
    int max_smooth_iter_int = 50;

    double smooth_size = 1;
    double edge_strech_threshold = 2.7;

    std::string fileprefix = "";
    bool use_binary = true;

    std::vector<int> crck_pts; // for debug only...

    std::vector<Eigen::Vector3d> vertices_copy; // a copy of all vertices before
        // smoothening.....for debug purpose..
    std::vector<Eigen::Vector3d>
        vertices_initial; // .................... after smoothening......

    void ProcessOutput(Mesh3D &newmesh); // Prepare for output...

    std::vector<Eigen::Vector4i> output_quad;
    std::vector<int> output_vert_flag;
    int num_of_surf_quad;

    void ParametersIn(std::string cmd_input_path);

    int clean_debris = 0; //clean debris if >0 serve as a threshold

    int pid_min = -1;
    int pid_max = -1;

    std::vector<Eigen::Vector4i> output_mesh_backup; //store a copy of boundary outputmesh
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> tris;
    std::string m_vtk_path;
};