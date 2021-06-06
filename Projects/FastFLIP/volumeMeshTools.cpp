#include "volumeMeshTools.h"
#include <fstream>
void vdbToolsWapper::writeObj(const std::string& objname, const std::vector<openvdb::Vec3f>& verts, const std::vector <openvdb::Vec4I>& faces)
{
    std::ofstream outfile(objname);
    //write vertices
    for (unsigned int i = 0; i < verts.size(); ++i)
        outfile << "v" << " " << verts[i][0] << " " << verts[i][1] << " " << verts[i][2] << std::endl;
    //write triangle faces
    for (unsigned int i = 0; i < faces.size(); ++i)
        outfile << "f" << " " << faces[i][3] + 1 << " " << faces[i][2] + 1 << " " << faces[i][1] + 1 << " " << faces[i][0] + 1 << std::endl;
    outfile.close();
}
openvdb::FloatGrid::Ptr vdbToolsWapper::readMeshToLevelset(const std::string& filename, float h)
{
    std::vector<LosTopos::Vec3f> vertList;
    std::vector<LosTopos::Vec3ui> faceList;
    std::vector<LosTopos::Vec4ui> quadList;
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open. Terminating.\n";
        exit(-1);
    }

    int ignored_lines = 0;
    std::string line;

    while (!infile.eof()) {
        std::getline(infile, line);
        if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vt") && line.substr(0, 2) != std::string("vn")) {
            std::stringstream data(line);
            char c;
            LosTopos::Vec3f point;
            data >> c >> point[0] >> point[1] >> point[2];
            vertList.push_back(point);
        }
        else if (line.substr(0, 1) == std::string("f")) {
            char c;
            int v0, v1, v2;
            std::stringstream data(line);
            std::string token;
            LosTopos::Vec3ui face_index;
            // throw away f
            std::getline(data, token, ' ');
            // read the face index of triangles
            int i = 0;
            std::vector<int> face_idx;
            while (std::getline(data, token, ' '))
            {
                face_idx.push_back(std::atoi(token.c_str()) - 1);
                i++;
            }
            if (i == 3)
            {
                LosTopos::Vec3ui tri_idx(face_idx[0], face_idx[1], face_idx[2]);
                faceList.push_back(tri_idx);
            }
            if (i == 4)
            {
                LosTopos::Vec4ui quad_idx(face_idx[0], face_idx[1], face_idx[2], face_idx[3]);
                quadList.push_back(quad_idx);
            }
        }
        else {
            ++ignored_lines;
        }
    }
    infile.close();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    points.resize(vertList.size());
    triangles.resize(faceList.size());
    quads.resize(quadList.size());
    tbb::parallel_for(0, (int)vertList.size(), 1, [&](int p)
        {
            points[p] = openvdb::Vec3s(vertList[p][0], vertList[p][1], vertList[p][2]);
        });
    tbb::parallel_for(0, (int)faceList.size(), 1, [&](int p)
        {
            triangles[p] = openvdb::Vec3I(faceList[p][0], faceList[p][1], faceList[p][2]);
        });
    tbb::parallel_for(0, (int)quadList.size(), 1, [&](int p)
        {
            quads[p] = openvdb::Vec4I(quadList[p][0], quadList[p][1], quadList[p][2], quadList[p][3]);
        });
    // TODO: need to find way to fill interior
    openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*openvdb::math::Transform::createLinearTransform(h), points, triangles, quads, 4, 4);
    return grid;
}

openvdb::FloatGrid::Ptr vdbToolsWapper::particleToLevelset(const std::vector<FLIP_particle>& particles, double radius, double voxelSize)
{
    MyParticleList pa(particles.size(), 1, 1);
    tbb::parallel_for(0, (int)particles.size(), 1, [&](int p)
        {
            LosTopos::Vec3f pos = particles[p].pos;
            pa.set(p, openvdb::Vec3R((double)(pos[0]), (double)(pos[1]), (double)(pos[2])), radius);
        });
    printf("%d,%d\n", (int)(pa.size()), (int)(particles.size()));
    //double voxelSize = radius / 1.001*2.0 / sqrt(3.0) / 2.0, halfWidth = 2.0;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, 3.0);
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32> raster(*ls);


    raster.setGrainSize(1);//a value of zero disables threading
    raster.rasterizeSpheres(pa);
    openvdb::CoordBBox bbox = pa.getBBox(*ls);
    std::cout << bbox.min() << std::endl;
    std::cout << bbox.max() << std::endl;
    raster.finalize(true);
    return ls;
}

void vdbToolsWapper::export_VDB(std::string path, int frame, const std::vector<FLIP_particle>& particles, double radius, std::vector<openvdb::Vec3s>& points, std::vector<openvdb::Vec3I>& triangles, std::shared_ptr<sparse_fluid8x8x8> eulerian_fluids)
{
    MyParticleList pa(particles.size(), 1, 1);
    tbb::parallel_for(0, (int)particles.size(), 1, [&](int p)
        {
            LosTopos::Vec3f pos = particles[p].pos;
            pa.set(p, openvdb::Vec3R((double)(pos[0]), (double)(pos[1]), (double)(pos[2])), radius);
        });
    printf("%d,%d\n", (int)(pa.size()), (int)(particles.size()));
    double voxelSize = radius / 1.001 * 2.0 / sqrt(3.0) / 2.0, halfWidth = 2.0;
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32> raster(*ls);


    raster.setGrainSize(1);//a value of zero disables threading
    raster.rasterizeSpheres(pa);
    openvdb::CoordBBox bbox = pa.getBBox(*ls);
    std::cout << bbox.min() << std::endl;
    std::cout << bbox.max() << std::endl;
    raster.finalize(true);

    std::vector<openvdb::Vec4I> quads;

    //        openvdb::tools::volumeToMesh(*ls, points, triangles, quads, 0.0, 0.0);
    openvdb::tools::volumeToMesh(*ls, points, quads, 0.0);
    printf("meshing done\n");

    std::ostringstream strout;
    strout << path << "/liquidmesh_" << std::setfill('0') << std::setw(5) << frame << ".obj";

    std::string filepath = strout.str();

    FILE* outfile;
    outfile=fopen(filepath.c_str(),"w");

    //write vertices
    for (unsigned int i = 0; i < points.size(); ++i) {
        fprintf(outfile, "v %e %e %e\n", points[i][0], points[i][1], points[i][2]);
    }
    //write quad face
    for (unsigned int i = 0; i < quads.size(); ++i) {
        fprintf(outfile, "f %d %d %d %d\n", quads[i][3] + 1, quads[i][2] + 1, quads[i][1] + 1, quads[i][0] + 1);
    }
    fclose(outfile);
}

void vdbToolsWapper::outputgeo(std::string path, int frame, const std::vector<FLIP_particle>& particles) {
    std::string filestr = path + std::string("/particle%04d.geo");
    char filename[1024];
    sprintf(filename, filestr.c_str(), frame);
    FILE* out_file = fopen(filename, "w");
    //header information;
    fprintf(out_file, "PGEOMETRY V5\n");
    fprintf(out_file, "NPoints %d NPrims 1\n", (int)particles.size());
    fprintf(out_file, "NPointGroups 0 NPrimGroups 0\n");
    fprintf(out_file, "NPointAttrib 0 NVertexAttrib 0 NPrimAttrib 1 NAttrib 0\n\n");

    for (int i = 0; i < particles.size(); i++) {
        float x = particles[i].pos[0];
        float y = particles[i].pos[1];
        float z = particles[i].pos[2];
        fprintf(out_file, "%f %f %f 1\n", x, y, z);
    }

    //end information
    fprintf(out_file, "PrimitiveAttrib\n generator 1 index 1 papi\n");
    fprintf(out_file, "Part %zd", particles.size());
    for (int i = 0; i < particles.size(); i++) {
        fprintf(out_file, " %d", i);
    }
    fprintf(out_file, " [0]\n");
    fprintf(out_file, "beginExtra \n");
    fprintf(out_file, "endExtra\n");
    fclose(out_file);
}

#ifdef USE_PARTIO
void vdbToolsWapper::outputBgeo(std::string path, int frame, const std::vector<LosTopos::Vec3f>& p_pos, const std::vector<LosTopos::Vec3f>& p_vel, const std::vector<float>& p_w, const std::vector<float>& p_v)
{
    std::string filestr = path + std::string("/particle%04d.bgeo");
    char filename[1024];
    sprintf(filename, filestr.c_str(), frame);
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute vH, posH, uwH, vwH;
    vH = parts->addAttribute("v", Partio::VECTOR, 3);
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    uwH = parts->addAttribute("uweight", Partio::VECTOR, 1);
    vwH = parts->addAttribute("vweight", Partio::VECTOR, 1);
    for (int i = 0; i < p_pos.size(); i++) {
        int idx = parts->addParticle();
        float* _p = parts->dataWrite<float>(posH, idx);
        float* _v = parts->dataWrite<float>(vH, idx);
        float* _uw = parts->dataWrite<float>(uwH, idx);
        float* _vw = parts->dataWrite<float>(vwH, idx);
        _p[0] = p_pos[i][0];
        _p[1] = p_pos[i][1];
        _p[2] = p_pos[i][2];
        _v[0] = p_vel[i][0];
        _v[1] = p_vel[i][1];
        _v[2] = p_vel[i][2];
        _uw[0] = p_w[i];
        _vw[0] = p_v[i];
    }
    Partio::write(filename, *parts);
    parts->release();
}
 void vdbToolsWapper::outputBgeo(std::string path, int frame, const std::vector<openvdb::Vec3s>& points, std::shared_ptr<sparse_fluid8x8x8> eulerian_fluids)
{
    std::string filestr = path + std::string("/test_/%04d.bgeo");
    char filename[1024];
    sprintf(filename, filestr.c_str(), frame);
    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    for (int i = 0; i < points.size(); i++) {
        int idx = parts->addParticle();
        float* _p = parts->dataWrite<float>(posH, idx);

        _p[0] = points[i][0];
        _p[1] = points[i][1];
        _p[2] = points[i][2];
        LosTopos::Vec3f pos(points[i][0], points[i][1], points[i][2]);

    }
    Partio::write(filename, *parts);
    parts->release();
}

void vdbToolsWapper::outputBgeo(std::string path, int frame, const std::vector<FLIP_particle>& flip_p)
{
     std::string filestr = path + std::string("/particle%04d.bgeo");
     char filename[1024];
     sprintf(filename, filestr.c_str(), frame);
     Partio::ParticlesDataMutable* parts = Partio::create();
     Partio::ParticleAttribute vH, posH;
     vH = parts->addAttribute("v", Partio::VECTOR, 3);
     posH = parts->addAttribute("position", Partio::VECTOR, 3);
     auto idx = parts->addParticles(flip_p.size());
     size_t i = 0;
     for (auto itr = idx; itr !=parts->end(); ++itr,++i) {
         //int idx = parts->addParticle();
         float* _p = parts->dataWrite<float>(posH, i);
         float* _v = parts->dataWrite<float>(vH, i);
         _p[0] = flip_p[i].pos[0];
         _p[1] = flip_p[i].pos[1];
         _p[2] = flip_p[i].pos[2];
         _v[0] = flip_p[i].vel[0];
         _v[1] = flip_p[i].vel[1];
         _v[2] = flip_p[i].vel[2];
     }
     
     Partio::write(filename, *parts,/*force compresse*/false);
     parts->release();
}
void outputBgeo(std::string path,  
const std::vector<glm::vec3> &pos, const std::vector<glm::vec3> &vel)
{
     Partio::ParticlesDataMutable* parts = Partio::create();
     Partio::ParticleAttribute vH, posH;
     vH = parts->addAttribute("v", Partio::VECTOR, 3);
     posH = parts->addAttribute("position", Partio::VECTOR, 3);
     auto idx = parts->addParticles(pos.size());
     size_t i = 0;
     for (auto itr = idx; itr !=parts->end(); ++itr,++i) {
         //int idx = parts->addParticle();
         float* _p = parts->dataWrite<float>(posH, i);
         float* _v = parts->dataWrite<float>(vH, i);
         _p[0] = pos[i][0];
         _p[1] = pos[i][1];
         _p[2] = pos[i][2];
         _v[0] = vel[i][0];
         _v[1] = vel[i][1];
         _v[2] = vel[i][2];
     }
     
     Partio::write(path.c_str(), *parts,/*force compresse*/false);
     parts->release();
}

#endif