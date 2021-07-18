#ifndef VOLUMETOOLS_H
#define VOLUMETOOLS_H
#include "Sparse_buffer.h"
#include "openvdb/tools/GridOperators.h"
#include "openvdb/tools/Interpolation.h"
#include "openvdb/tools/LevelSetFilter.h"
#include "openvdb/tools/LevelSetPlatonic.h"
#include "openvdb/tools/LevelSetSphere.h"
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tools/ParticlesToLevelSet.h"
#include "openvdb/tools/VolumeToMesh.h"
#include "vec.h"
#include <Partio.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/util/Util.h>
#include <vector>





using namespace Partio;
class MyParticleList {

protected:
  struct MyParticle {

    openvdb::Vec3R p, v;
    openvdb::Real r;
    MyParticle() {
      p = openvdb::Vec3R(0);
      v = openvdb::Vec3R(0);
      r = 0;
    }
    MyParticle(const MyParticle &_p) {
      p = _p.p;
      v = _p.v;
      r = _p.r;
    }
  };
  openvdb::Real mRadiusScale;
  openvdb::Real mVelocityScale;
  std::vector<MyParticle> mParticleList;

public:
  typedef openvdb::Vec3R PosType;
  typedef openvdb::Vec3R value_type;

  MyParticleList(size_t size, openvdb::Real rScale = 1,
                 openvdb::Real vScale = 1)
      : mRadiusScale(rScale), mVelocityScale(vScale) {
    mParticleList.resize(size);
  }
  MyParticleList(openvdb::Real rScale = 1, openvdb::Real vScale = 1)
      : mRadiusScale(rScale), mVelocityScale(vScale) {
    mParticleList.resize(0);
  }
  void free() { mParticleList = std::vector<MyParticle>(0); }
  void set(int i, const openvdb::Vec3R &p, const openvdb::Real &r,
           const openvdb::Vec3R &v = openvdb::Vec3R(0, 0, 0)) {
    MyParticle pa;
    pa.p = p;
    pa.r = r;
    pa.v = v;
    mParticleList[i] = pa;
  }
  void add(const openvdb::Vec3R &p, const openvdb::Real &r,
           const openvdb::Vec3R &v = openvdb::Vec3R(0, 0, 0)) {
    MyParticle pa;
    pa.p = p;
    pa.r = r;
    pa.v = v;
    mParticleList.push_back(pa);
  }
  /// @return coordinate bbox in the space of the specified transfrom
  openvdb::CoordBBox getBBox(const openvdb::GridBase &grid) {
    openvdb::CoordBBox bbox;
    openvdb::Coord &min = bbox.min(), &max = bbox.max();
    openvdb::Vec3R pos;
    openvdb::Real rad, invDx = 1.0 / grid.voxelSize()[0];
    for (size_t n = 0, e = this->size(); n < e; ++n) {
      this->getPosRad(n, pos, rad);
      const openvdb::Vec3d xyz = grid.worldToIndex(pos);
      const openvdb::Real r = rad * invDx;
      for (int i = 0; i < 3; ++i) {
        min[i] = openvdb::math::Min(min[i], openvdb::math::Floor(xyz[i] - r));
        max[i] = openvdb::math::Max(max[i], openvdb::math::Ceil(xyz[i] + r));
      }
    }
    return bbox;
  }
  // typedef int AttributeType;
  // The methods below are only required for the unit-tests
  openvdb::Vec3R pos(int n) const { return mParticleList[n].p; }
  openvdb::Vec3R vel(int n) const {
    return mVelocityScale * mParticleList[n].v;
  }
  openvdb::Real radius(int n) const {
    return mRadiusScale * mParticleList[n].r;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// The methods below are the only ones required by tools::ParticleToLevelSet
  /// @note We return by value since the radius and velocities are modified
  /// by the scaling factors! Also these methods are all assumed to
  /// be thread-safe.

  /// Return the total number of particles in list.
  ///  Always required!
  size_t size() const { return mParticleList.size(); }

  /// Get the world space position of n'th particle.
  /// Required by ParticledToLevelSet::rasterizeSphere(*this,radius).
  void getPos(size_t n, openvdb::Vec3R &pos) const { pos = mParticleList[n].p; }

  void getPosRad(size_t n, openvdb::Vec3R &pos, openvdb::Real &rad) const {
    pos = mParticleList[n].p;
    rad = mRadiusScale * mParticleList[n].r;
  }
  void getPosRadVel(size_t n, openvdb::Vec3R &pos, openvdb::Real &rad,
                    openvdb::Vec3R &vel) const {
    pos = mParticleList[n].p;
    rad = mRadiusScale * mParticleList[n].r;
    vel = mVelocityScale * mParticleList[n].v;
  }
  // The method below is only required for attribute transfer
  void getAtt(size_t n, openvdb::Index32 &att) const {
    att = openvdb::Index32(n);
  }
};
namespace vdbToolsWapper {
  static void writeObj(const std::string &objname,
                       const std::vector<openvdb::Vec3f> &verts,
                       const std::vector<openvdb::Vec4I> &faces) {
    ofstream outfile(objname);

    // write vertices
    for (unsigned int i = 0; i < verts.size(); ++i)
      outfile << "v"
              << " " << verts[i][0] << " " << verts[i][1] << " " << verts[i][2]
              << std::endl;
    // write triangle faces
    for (unsigned int i = 0; i < faces.size(); ++i)
      outfile << "f"
              << " " << faces[i][3] + 1 << " " << faces[i][2] + 1 << " "
              << faces[i][1] + 1 << " " << faces[i][0] + 1 << std::endl;
    outfile.close();
  }

  static openvdb::FloatGrid::Ptr readMeshToLevelset(const std::string &filename,
                                                    float h) {
    std::vector<FLUID::Vec3f> vertList;
    std::vector<FLUID::Vec3ui> faceList;
    std::ifstream infile(filename);
    if (!infile) {
      std::cerr << "Failed to open. Terminating.\n";
      exit(-1);
    }

    int ignored_lines = 0;
    std::string line;

    while (!infile.eof()) {
      std::getline(infile, line);
      if (line.substr(0, 1) == std::string("v")) {
        std::stringstream data(line);
        char c;
        FLUID::Vec3f point;
        data >> c >> point[0] >> point[1] >> point[2];
        vertList.push_back(point);
      } else if (line.substr(0, 1) == std::string("f")) {
        std::stringstream data(line);
        char c;
        int v0, v1, v2;
        data >> c >> v0 >> v1 >> v2;
        faceList.push_back(FLUID::Vec3ui(v0 - 1, v1 - 1, v2 - 1));
      } else {
        ++ignored_lines;
      }
    }
    infile.close();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    points.resize(vertList.size());
    triangles.resize(faceList.size());
    tbb::parallel_for(0, (int)vertList.size(), 1, [&](int p) {
      points[p] =
          openvdb::Vec3s(vertList[p][0], vertList[p][1], vertList[p][2]);
    });
    tbb::parallel_for(0, (int)faceList.size(), 1, [&](int p) {
      triangles[p] =
          openvdb::Vec3I(faceList[p][0], faceList[p][1], faceList[p][2]);
    });
    openvdb::FloatGrid::Ptr grid =
        openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
            *openvdb::math::Transform::createLinearTransform(h), points,
            triangles, 3.0);
    return grid;
  }
  static openvdb::FloatGrid::Ptr
  particleToLevelset(const std::vector<FLIP_particle> &particles,
                     double radius) {
    MyParticleList pa(particles.size(), 1, 1);
    tbb::parallel_for(0, (int)particles.size(), 1, [&](int p) {
      FLUID::Vec3f pos = particles[p].pos;
      pa.set(
          p,
          openvdb::Vec3R((double)(pos[0]), (double)(pos[1]), (double)(pos[2])),
          radius);
    });
    printf("%d,%d\n", (int)(pa.size()), (int)(particles.size()));
    double voxelSize = radius / 1.001 * 2.0 / sqrt(3.0) / 2.0, halfWidth = 2.0;
    openvdb::FloatGrid::Ptr ls =
        openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, 4.0);
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32>
        raster(*ls);

    raster.setGrainSize(1); // a value of zero disables threading
    raster.rasterizeSpheres(pa);
    openvdb::CoordBBox bbox = pa.getBBox(*ls);
    std::cout << bbox.min() << std::endl;
    std::cout << bbox.max() << std::endl;
    raster.finalize(true);
    return ls;
  }
//  static void export_VDB_Mesh(std::string path, int frame,
//                              const std::vector<FLIP_particle> &particles,
//                              double radius,
//                              std::vector<openvdb::Vec3s> &points,
//                              std::vector<openvdb::Vec3I> &triangles,
//                              sparse_fluid8x8x8 &eulerian_fluids) {
//      using namespace std;
//    MyParticleList pa(particles.size(), 1, 1);
//    tbb::parallel_for(0, (int)particles.size(), 1, [&](int p) {
//      FLUID::Vec3f pos = particles[p].pos;
//      pa.set(
//          p,
//          openvdb::Vec3R((double)(pos[0]), (double)(pos[1]), (double)(pos[2])),
//          radius);
//    });
//    printf("%d,%d\n", (int)(pa.size()), (int)(particles.size()));
//    double voxelSize = radius / 1.001 * 2.0 / sqrt(3.0) / 2.0, halfWidth = 2.0;
//    openvdb::FloatGrid::Ptr ls =
//        openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
//    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32>
//        raster(*ls);
//
//    raster.setGrainSize(1); // a value of zero disables threading
//    raster.rasterizeSpheres(pa);
//    openvdb::CoordBBox bbox = pa.getBBox(*ls);
//    std::cout << bbox.min() << std::endl;
//    std::cout << bbox.max() << std::endl;
//    raster.finalize(true);
//
//    std::vector<openvdb::Vec4I> quads;
//
//    openvdb::tools::volumeToMesh(*ls, points, triangles, quads, 0.0, 0.0);
//    // openvdb::tools::volumeToMesh(*ls, points, quads, 0.0);
//    printf("meshing done\n");
//
//    ostringstream strout;
//    strout << path << "/liquidmesh_" << setfill('0') << setw(5) << frame
//           << ".obj";
//
//    std::string filepath = strout.str();
//
//    ofstream outfile(filepath.c_str());
//
//    // write vertices
//    for (unsigned int i = 0; i < points.size(); ++i)
//      outfile << "v"
//              << " " << points[i][0] << " " << points[i][1] << " "
//              << points[i][2] << std::endl;
//
//    for (auto t : triangles) {
//      outfile << "v"
//              << " " << t[0] + 1 << " " << t[1] + 1 << " " << t[2] + 1
//              << std::endl;
//    }
//    for (auto t : quads) {
//      outfile << "v"
//              << " " << t[0] + 1 << " " << t[1] + 1 << " " << t[2] + 1 << " "
//              << t[3] + 1 << std::endl;
//    }
//
//    outfile.close();
//  }
//
//  static void export_VDB(std::string path, int frame,
//                         const std::vector<FLIP_particle> &particles,
//                         double radius, std::vector<openvdb::Vec3s> &points,
//                         std::vector<openvdb::Vec3I> &triangles,
//                         sparse_fluid8x8x8 eulerian_fluids) {
//      using namespace std;
//    MyParticleList pa(particles.size(), 1, 1);
//    tbb::parallel_for(0, (int)particles.size(), 1, [&](int p) {
//      FLUID::Vec3f pos = particles[p].pos;
//      pa.set(
//          p,
//          openvdb::Vec3R((double)(pos[0]), (double)(pos[1]), (double)(pos[2])),
//          radius);
//    });
//    printf("%d,%d\n", (int)(pa.size()), (int)(particles.size()));
//    double voxelSize = radius / 1.001 * 2.0 / sqrt(3.0) / 2.0, halfWidth = 2.0;
//    openvdb::FloatGrid::Ptr ls =
//        openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
//    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32>
//        raster(*ls);
//
//    raster.setGrainSize(1); // a value of zero disables threading
//    raster.rasterizeSpheres(pa);
//    openvdb::CoordBBox bbox = pa.getBBox(*ls);
//    std::cout << bbox.min() << std::endl;
//    std::cout << bbox.max() << std::endl;
//    raster.finalize(true);
//
//    std::vector<openvdb::Vec4I> quads;
//
//    //        openvdb::tools::volumeToMesh(*ls, points, triangles, quads, 0.0,
//    //        0.0);
//    openvdb::tools::volumeToMesh(*ls, points, quads, 0.0);
//    printf("meshing done\n");
//
//    ostringstream strout;
//    strout << path << "/liquidmesh_" << setfill('0') << setw(5) << frame
//           << ".obj";
//
//    string filepath = strout.str();
//
//    ofstream outfile(filepath.c_str());
//
//    // write vertices
//    for (unsigned int i = 0; i < points.size(); ++i)
//      outfile << "v"
//              << " " << points[i][0] << " " << points[i][1] << " "
//              << points[i][2] << std::endl;
//
//    // write quad face
//    //        for (unsigned int i = 0; i < quads.size(); ++i)
//    //            outfile << "f" << " " << quads[i][3] + 1 << " " << quads[i][2]
//    //            + 1 << " " << quads[i][1] + 1 << " " << quads[i][0] + 1 <<
//    //            std::endl;
//    for (unsigned int i = 0; i < quads.size(); ++i) {
//      triangles.push_back(
//          openvdb::Vec3I(quads[i][3] + 1, quads[i][2] + 1, quads[i][1] + 1));
//      triangles.push_back(
//          openvdb::Vec3I(quads[i][3] + 1, quads[i][1] + 1, quads[i][0] + 1));
//      openvdb::Vec3s p1 =
//          (points[quads[i][3]] + points[quads[i][2]] + points[quads[i][1]]) /
//          3.;
//      openvdb::Vec3s p2 =
//          (points[quads[i][3]] + points[quads[i][1]] + points[quads[i][0]]) /
//          3.;
//      FLUID::Vec3f P1(p1[0], p1[1], p1[2]);
//      FLUID::Vec3f P2(p2[0], p2[1], p2[2]);
//      if ((abs(eulerian_fluids.get_solid_phi(P1)) <
//           abs(eulerian_fluids.get_liquid_phi(P1))) ||
//          eulerian_fluids.get_solid_phi(P1) < 0) {
//        outfile << "f"
//                << " " << quads[i][3] + 1 << " " << quads[i][2] + 1 << " "
//                << quads[i][1] + 1 << " solid" << std::endl;
//        //                outfile << "f" << " " << quads[i][3] + 1 << " " <<
//        //                quads[i][2] + 1 << " " << quads[i][1] + 1 <<
//        //                std::endl;
//      } else {
//        outfile << "f"
//                << " " << quads[i][3] + 1 << " " << quads[i][2] + 1 << " "
//                << quads[i][1] + 1 << " fluid" << std::endl;
//        //                outfile << "f" << " " << quads[i][3] + 1 << " " <<
//        //                quads[i][2] + 1 << " " << quads[i][1] + 1 <<
//        //                std::endl;
//      }
//      if ((abs(eulerian_fluids.get_solid_phi(P2)) <
//           abs(eulerian_fluids.get_liquid_phi(P2))) ||
//          eulerian_fluids.get_solid_phi(P2) < 0) {
//        outfile << "f"
//                << " " << quads[i][3] + 1 << " " << quads[i][1] + 1 << " "
//                << quads[i][0] + 1 << " solid" << std::endl;
//        //                outfile << "f" << " " << quads[i][3] + 1 << " " <<
//        //                quads[i][1] + 1 << " " << quads[i][0] + 1 <<
//        //                std::endl;
//      } else {
//        outfile << "f"
//                << " " << quads[i][3] + 1 << " " << quads[i][1] + 1 << " "
//                << quads[i][0] + 1 << " fluid" << std::endl;
//        //                outfile << "f" << " " << quads[i][3] + 1 << " " <<
//        //                quads[i][1] + 1 << " " << quads[i][0] + 1 <<
//        //                std::endl;
//      }
//    }
//
//    for (unsigned int i = 0; i < points.size(); i++) {
//      FLUID::Vec3f pos(points[i][0], points[i][1], points[i][2]);
//      FLUID::Vec3f vel = eulerian_fluids.get_velocity(pos);
//      outfile << "vn"
//              << " " << vel[0] << " " << vel[1] << " " << vel[2] << std::endl;
//    }
//
//    outfile.close();
//  }

  static void genVDBVelGrid(std::vector<FLUID::Vec3i> &voxels, std::vector<FLUID::Vec3f> &vels, double h, FLUID::Vec3f offset, openvdb::GridPtrVec &grids)
  {

      std::vector<std::string> vel_names(0);
      vel_names.push_back("u");
      vel_names.push_back("v");
      vel_names.push_back("w");
      for(int channel=0;channel<3;channel++)
      {
          openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
          openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
          int total = 0;
          for(int i=0;i<voxels.size();i++)
          {

              openvdb::math::Coord xyz(voxels[i][0],voxels[i][1],voxels[i][2]);
              accessor.setValue(xyz, vels[i][channel]);
              total += 1;

          }

          //std::cout << "[ Valid vdb voxel: " << total << " ] " << std::endl << std::endl;

          const openvdb::math::Vec3d offset(offset[0],offset[1],offset[2]);
          openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(h);
          transform->postTranslate(offset);
          grid->setTransform(transform);
          grid->setGridClass(openvdb::GRID_FOG_VOLUME);
          grid->setName(vel_names[channel]);

          grids.push_back(grid);
      }
  }

  static void outputBin(std::string path, int frame,  const std::vector<FLIP_particle> &particles)
  {
      std::string filestr = path + std::string("\%04d.bin");
      uint64_t N = particles.size();

      char filename[1024];
      sprintf(filename, filestr.c_str(), frame);

      float* pos = new float[4*N];
      float* vel = new float[4*N];
      for(uint64_t i=0; i<particles.size(); i++)
      {
          pos[i*4 + 0] = particles[i].pos[0];
          pos[i*4 + 1] = particles[i].pos[1];
          pos[i*4 + 2] = particles[i].pos[2];
          pos[i*4 + 3] = 1.0f;

          vel[i*4 + 0] = particles[i].vel[0];
          vel[i*4 + 1] = particles[i].vel[1];
          vel[i*4 + 2] = particles[i].vel[2];
          vel[i*4 + 3] = 0.0f;
      }

      FILE *data_file = fopen(filename,"wb");
      fwrite(&N, sizeof(uint64_t),1, data_file);
      fwrite(pos,4*sizeof(float),N,data_file);
      fwrite(vel,4*sizeof(float),N,data_file);
      fclose(data_file);
      printf("timestep %d done\n",frame);
      delete[]pos;
      delete[]vel;

  }
/*
  static void readBgeo(std::string file, uint64_t &num, std::vector<float> &pos, std::vector<float> &vel)
  {
      using namespace Partio;
    ParticleData* simple=Partio::read(file.c_str());
    if(!simple) return 0;

    // prepare iteration
    auto iterator=simple->begin(), end=simple->end();
    ParticleAttribute posAttr;
    if(!simple->attributeInfo("position",attr) || !simple->attributeInfo("v",attr) || attr.type != VECTOR || attr.count != 3)
    {
      return 0;
    }

    ParticleAccessor posAcc;
    iterator.addAccessor(posAcc);

    // compute sum
    uint64_t cnt = 0;
    for (auto it=simple->begin(); it!=simple->end(); ++it) {
        float* data = posAcc.raw<float>(it);
        cnt++;
    }
    num = cnt;
    std::cout<<"num Particles:"<<num<<std::endl;
    std::cout<<"num Particles:"<<(int)(simple->end() - simple->begin())<<std::endl;
    pos.reserve(4*num); vel.reserve(4*num);
    for (auto it=simple->begin(); it!=simple->end(); ++it) {
        float* data = posAcc.raw<float>(it);
        pos.emplace_back(data[0]);
        pos.emplace_back(data[1]);
        pos.emplace_back(data[2]);
        pos.emplace_back(1.0f);
        vel.emplace_back(data[3]);
        vel.emplace_back(data[4]);
        vel.emplace_back(data[5]);
        vel.emplace_back(0.0f);
    }

    simple->release();

  }
*/
  static void readBin(std::string file, uint64_t & num, std::vector<float> &pos, std::vector<float> &vel)
  {
      FILE *data_file = fopen(file.c_str(), "rb");
      uint64_t num_of_particles;
      fread(&num_of_particles, sizeof(uint64_t), 1, data_file);
      num = num_of_particles;
      std::cout<<"num Particles:"<<num<<std::endl;
      pos.resize(num*4);
      vel.resize(num*4);
      fread(&(pos[0]), 4*sizeof(float), num, data_file);
      fread(&(vel[0]), 4*sizeof(float), num, data_file);
      fclose(data_file);
  }


  static void outputBgeo(std::string path, int frame,
                         const std::vector<FLIP_particle> &particles) {
    std::string filestr = path + std::string("\%04d.bgeo");
    char filename[1024];
    sprintf(filename, filestr.c_str(), frame);
    Partio::ParticlesDataMutable *parts = Partio::create();
    Partio::ParticleAttribute vH, posH;
    vH = parts->addAttribute("v", Partio::VECTOR, 3);
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    for (int i = 0; i < particles.size(); i++) {
      int idx = parts->addParticle();
      float *_p = parts->dataWrite<float>(posH, idx);
      float *_pv = parts->dataWrite<float>(vH, idx);
      _p[0] = particles[i].pos[0];
      _p[1] = particles[i].pos[1];
      _p[2] = particles[i].pos[2];
      _pv[0] = particles[i].vel[0];
      _pv[1] = particles[i].vel[1];
      _pv[2] = particles[i].vel[2];
    }
    Partio::write(filename, *parts);
    parts->release();
  }

//  static void outputBgeo(std::string path, int frame,
//                         const std::vector<openvdb::Vec3s> &points,
//                         sparse_fluid8x8x8 eulerian_fluids) {
//    std::string filestr = path + std::string("test_\%04d.bgeo");
//    char filename[1024];
//    sprintf(filename, filestr.c_str(), frame);
//    Partio::ParticlesDataMutable *parts = Partio::create();
//    Partio::ParticleAttribute cH, posH;
//    cH = parts->addAttribute("color", Partio::VECTOR, 1);
//    posH = parts->addAttribute("position", Partio::VECTOR, 3);
//    for (int i = 0; i < points.size(); i++) {
//      int idx = parts->addParticle();
//      float *_p = parts->dataWrite<float>(posH, idx);
//      float *_c = parts->dataWrite<float>(cH, idx);
//      _p[0] = points[i][0];
//      _p[1] = points[i][1];
//      _p[2] = points[i][2];
//      FLUID::Vec3f pos(points[i][0], points[i][1], points[i][2]);
//      if ((abs(eulerian_fluids.get_solid_phi(pos)) <
//           abs(eulerian_fluids.get_liquid_phi(pos))) ||
//          eulerian_fluids.get_solid_phi(pos) < 0)
//        _c[0] = 1;
//      else
//        _c[0] = -1;
//    }
//    Partio::write(filename, *parts);
//    parts->release();
//  }
};
#endif // VOLUMETOOLS_H
