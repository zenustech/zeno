#ifndef __SCENE_PARSER_H_
#define __SCENE_PARSER_H_

#include <cxxopts.hpp>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fstream>

#include <filesystem>
#include <string>
namespace fs = std::filesystem;

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
namespace rj = rapidjson;

#include "../FLIP/fluidsim.h"
#include "../FLIP/volumeMeshTools.h"
#include "../OverLaySolver/OverLaySolver.h"
using GridT =
    typename openvdb::Grid<typename openvdb::tree::Tree4<float, 5, 4, 3>::Type>;
using TreeT = typename GridT::TreeType;

inline struct {
  float dx;
  float dt;
  float T;
  float g=-9.81;
  std::string outputDir;
  std::vector<std::string> boundaryFiles;
} simConfigs;

inline FLUID::Vec3f offset;
inline typename GridT::Ptr model;
inline std::vector<typename GridT::Ptr> boundaries;

auto readPhiFromVdbFile(const std::string &fn) {
  typename GridT::Ptr grid;
  openvdb::io::File file(fn);
  file.open();
  openvdb::GridPtrVecPtr my_grids = file.getGrids();
  file.close();
  int count = 0;
  for (openvdb::GridPtrVec::iterator iter = my_grids->begin();
       iter != my_grids->end(); ++iter) {
    openvdb::GridBase::Ptr it = *iter;
    if ((*iter)->isType<GridT>()) {
      grid = openvdb::gridPtrCast<GridT>(*iter);
      count++;
      /// display meta data
      for (openvdb::MetaMap::MetaIterator it = grid->beginMeta();
           it != grid->endMeta(); ++it) {
        const std::string &name = it->first;
        openvdb::Metadata::Ptr value = it->second;
        std::string valueAsString = value->str();
        std::cout << name << " = " << valueAsString << std::endl;
      }
    }
  }
  return grid;
}

float sample_phi(const FLUID::Vec3f &position) {
  openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
      model->constTree(), model->transform());
  openvdb::math::Vec3<float> P(position[0], position[1], position[2]);
  return interpolator.wsSample(P); // ws denotes world space
}

float sphere_phi(const FLUID::Vec3f &position, const FLUID::Vec3f &centre,
                 float radius) {
  return dist(position, centre) - radius;
}
float cuboid_phi(const FLUID::Vec3f &position, const FLUID::Vec3f &bmin,
                 const FLUID::Vec3f &bmax) {
  // return dist(position, centre) - radius;
  for (int d = 0; d < 3; ++d)
    if (position[d] < bmin[d] || position[d] > bmax[d])
      return 1.f;
  return -1.f;
}

void parse_scene(std::string fn, FluidSim &simulator) {
  fs::path p{fn};
  if (p.empty())
    fmt::print("file not exist {}\n", fn);
  else {
    std::size_t size = fs::file_size(p);
    std::string configs;
    configs.resize(size);

    std::ifstream istrm(fn);
    if (!istrm.is_open())
      fmt::print("cannot open file {}\n", fn);
    else
      istrm.read(const_cast<char *>(configs.data()), configs.size());
    istrm.close();
    fmt::print("load the scene file of size {}\n", size);

    rj::Document doc;
    doc.Parse(configs.data());
    /// simulation
    if (auto it = doc.FindMember("simulation"); it != doc.MemberEnd())
      if (auto &sim = it->value; sim.IsObject()) {
        simConfigs.dx = sim["dx"].GetFloat();
        simConfigs.dt = sim["dt"].GetFloat();
        simConfigs.T  = sim["T"].GetFloat();
        simConfigs.g  = sim["G"].GetFloat();
        simConfigs.outputDir = sim["output_dir"].GetString();
        fmt::print(fg(fmt::color::cyan), "simulation: dx[{}], outputDir[{}]\n",
                   simConfigs.dx, simConfigs.outputDir);

        fmt::print("Initializing data\n");
        simulator.initialize(simConfigs.dx);
        simulator.eulerian_fluids.SetName(std::to_string(simConfigs.dx));
        simulator.setGravity(simConfigs.g);
      }

    fmt::print("Initializing liquid\n");
    if (auto it = doc.FindMember("models"); it != doc.MemberEnd())
      if (it->value.IsArray())
      {
        fmt::print(fg(fmt::color::cyan), "{} models in total\n",
                   it->value.Size());
        for (auto &particles : it->value.GetArray())
        {
          if (particles["format"] == "vdb")
          {
              FLUID::Vec3f offset;
              std::string fn = particles["file"].GetString();
              for (int d = 0; d < 3; ++d)
                  offset[d] = particles["offset"].GetArray()[d].GetFloat();
              model = readPhiFromVdbFile(fn);
              openvdb::CoordBBox box = model->evalActiveVoxelBoundingBox();
              auto world_min = model->indexToWorld(box.min());
              auto world_max = model->indexToWorld(box.max());
              simulator.set_liquid(
                      FLUID::Vec3f(world_min[0], world_min[1], world_min[2]) + offset,
                      FLUID::Vec3f(world_max[0], world_max[1], world_max[2]) + offset,
                      [offset](const FLUID::Vec3f &position) {
                          return sample_phi(position - offset);
                      });

          }
          else if (particles["format"] == "obj")
          {
              FLUID::Vec3f offset;
              std::string fn = particles["file"].GetString();
              for (int d = 0; d < 3; ++d)
                  offset[d] = particles["offset"].GetArray()[d].GetFloat();
              model = vdbToolsWapper::readMeshToLevelset(fn, simulator.dx);
              openvdb::CoordBBox box = model->evalActiveVoxelBoundingBox();
              auto world_min = model->indexToWorld(box.min());
              auto world_max = model->indexToWorld(box.max());
              simulator.set_liquid(
                      FLUID::Vec3f(world_min[0], world_min[1], world_min[2]) + offset,
                      FLUID::Vec3f(world_max[0], world_max[1], world_max[2]) + offset,
                      [offset](const FLUID::Vec3f &position) {
                          return sample_phi(position - offset);
                      });

          }
          else if (particles["format"] == "analytic")
          {
            if (particles["shape"] == "cuboid")
            {
              FLUID::Vec3f mi, ma;
              for (int d = 0; d < 3; ++d)
                mi[d] = particles["min"].GetArray()[d].GetFloat(),
                ma[d] = particles["max"].GetArray()[d].GetFloat();
              simulator.set_liquid(mi, ma,
                                   [&mi, &ma](const FLUID::Vec3f &position) {
                                     return cuboid_phi(position, mi, ma);
                                   });
              fmt::print("sampling cuboid: [{}, {}, {}] - [{}, {}, {}]\n",
                         mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]);
            }
            else if (particles["shape"] == "sphere")
            {
              FLUID::Vec3f center;
              float radius;
              radius = particles["radius"].GetFloat();
              FLUID::Vec3f r{radius, radius, radius};
              for (int d = 0; d < 3; ++d)
                center[d] = particles["center"].GetArray()[d].GetFloat();
              simulator.set_liquid(
                  center - r, center + r,
                  [&center, &radius](const FLUID::Vec3f &position) {
                    return sphere_phi(position, center, radius);
                  });
              fmt::print("sampling sphere: [{}, {}, {}] - {}\n", center[0],
                         center[1], center[2], radius);
            }
          }
        }
      }
    fmt::print("particles: {}\n", simulator.particles.size());

      if (auto it = doc.FindMember("emitters"); it != doc.MemberEnd())
      {
          if (it->value.IsArray()) {
              fmt::print(fg(fmt::color::cyan), "{} emitters in total\n",
                         it->value.Size());
              for (auto &particles : it->value.GetArray()) {
                  if (particles["format"] == "vdb") {
                      FLUID::Vec3f offset;
                      std::string fn = particles["file"].GetString();
                      for (int d = 0; d < 3; ++d)
                          offset[d] = particles["vel"].GetArray()[d].GetFloat();

                      model = readPhiFromVdbFile(fn);
                      openvdb::CoordBBox box = model->evalActiveVoxelBoundingBox();
                      auto world_min = model->indexToWorld(box.min());
                      auto world_max = model->indexToWorld(box.max());
                      simulator.set_liquid(
                              FLUID::Vec3f(world_min[0], world_min[1], world_min[2]),
                              FLUID::Vec3f(world_max[0], world_max[1], world_max[2]),
                              [offset](const FLUID::Vec3f &position) {
                                  return sample_phi(position);
                              });
                      simulator.addEmitter(offset, model);
                  } else if (particles["format"] == "obj") {
                      FLUID::Vec3f offset;
                      std::string fn = particles["file"].GetString();
                      for (int d = 0; d < 3; ++d)
                          offset[d] = particles["vel"].GetArray()[d].GetFloat();
                      model = vdbToolsWapper::readMeshToLevelset(fn, simulator.dx);
                      openvdb::CoordBBox box = model->evalActiveVoxelBoundingBox();
                      auto world_min = model->indexToWorld(box.min());
                      auto world_max = model->indexToWorld(box.max());
                      simulator.set_liquid(
                              FLUID::Vec3f(world_min[0], world_min[1], world_min[2]),
                              FLUID::Vec3f(world_max[0], world_max[1], world_max[2]),
                              [offset](const FLUID::Vec3f &position) {
                                  return sample_phi(position);
                              });
                      simulator.addEmitter(offset, model);
                  }
//                  else if (particles["format"] == "analytic")
//                  {
//                      if (particles["shape"] == "cuboid") {
//                          FLUID::Vec3f mi, ma;
//                          for (int d = 0; d < 3; ++d)
//                              mi[d] = particles["min"].GetArray()[d].GetFloat(),
//                                      ma[d] = particles["max"].GetArray()[d].GetFloat();
//                          simulator.set_liquid(mi, ma,
//                                               [&mi, &ma](const FLUID::Vec3f &position) {
//                                                   return cuboid_phi(position, mi, ma);
//                                               });
//                          fmt::print("sampling cuboid: [{}, {}, {}] - [{}, {}, {}]\n",
//                                     mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]);
//                      } else if (particles["shape"] == "sphere") {
//                          FLUID::Vec3f center;
//                          float radius;
//                          radius = particles["radius"].GetFloat();
//                          FLUID::Vec3f r{radius, radius, radius};
//                          for (int d = 0; d < 3; ++d)
//                              center[d] = particles["center"].GetArray()[d].GetFloat();
//                          simulator.set_liquid(
//                                  center - r, center + r,
//                                  [&center, &radius](const FLUID::Vec3f &position) {
//                                      return sphere_phi(position, center, radius);
//                                  });
//                          fmt::print("sampling sphere: [{}, {}, {}] - {}\n", center[0],
//                                     center[1], center[2], radius);
//                      }
//                  }
              }
          }
      }
      if (auto it = doc.FindMember("regions"); it != doc.MemberEnd())
      {
          if (it->value.IsArray()) {
              fmt::print(fg(fmt::color::cyan), "{} regions in total\n",
                         it->value.Size());
              for (auto &particles : it->value.GetArray()) {
                  if (particles["format"] == "vdb") {
                      std::string fn = particles["file"].GetString();
                      model = readPhiFromVdbFile(fn);
                      simulator.fluidDomains.push_back(model);
                      if( particles["fill"] == "true")
                      {
                          openvdb::CoordBBox box = model->evalActiveVoxelBoundingBox();
                          auto world_min = model->indexToWorld(box.min());
                          auto world_max = model->indexToWorld(box.max());
                          simulator.set_liquid(
                                  FLUID::Vec3f(world_min[0], world_min[1], world_min[2]),
                                  FLUID::Vec3f(world_max[0], world_max[1], world_max[2]),
                                  [&](const FLUID::Vec3f &position) {
                                      return sample_phi(position);
                                  });
                          simulator.fillRegion = true;
                      }
                  } else if (particles["format"] == "obj") {
                      std::string fn = particles["file"].GetString();
                      model =  vdbToolsWapper::readMeshToLevelset(fn, simulator.dx);
                      simulator.fluidDomains.push_back(model);
                      if( particles["fill"] == "true")
                      {
                          openvdb::CoordBBox box = model->evalActiveVoxelBoundingBox();
                          auto world_min = model->indexToWorld(box.min());
                          auto world_max = model->indexToWorld(box.max());
                          simulator.set_liquid(
                                  FLUID::Vec3f(world_min[0], world_min[1], world_min[2]),
                                  FLUID::Vec3f(world_max[0], world_max[1], world_max[2]),
                                  [&](const FLUID::Vec3f &position) {
                                      return sample_phi(position);
                                  });
                          simulator.fillRegion = true;
                      }
                  }


//                  else if (particles["format"] == "analytic")
//                  {
//                      if (particles["shape"] == "cuboid") {
//                          FLUID::Vec3f mi, ma;
//                          for (int d = 0; d < 3; ++d)
//                              mi[d] = particles["min"].GetArray()[d].GetFloat(),
//                                      ma[d] = particles["max"].GetArray()[d].GetFloat();
//                          simulator.set_liquid(mi, ma,
//                                               [&mi, &ma](const FLUID::Vec3f &position) {
//                                                   return cuboid_phi(position, mi, ma);
//                                               });
//                          fmt::print("sampling cuboid: [{}, {}, {}] - [{}, {}, {}]\n",
//                                     mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]);
//                      } else if (particles["shape"] == "sphere") {
//                          FLUID::Vec3f center;
//                          float radius;
//                          radius = particles["radius"].GetFloat();
//                          FLUID::Vec3f r{radius, radius, radius};
//                          for (int d = 0; d < 3; ++d)
//                              center[d] = particles["center"].GetArray()[d].GetFloat();
//                          simulator.set_liquid(
//                                  center - r, center + r,
//                                  [&center, &radius](const FLUID::Vec3f &position) {
//                                      return sphere_phi(position, center, radius);
//                                  });
//                          fmt::print("sampling sphere: [{}, {}, {}] - {}\n", center[0],
//                                     center[1], center[2], radius);
//                      }
//                  }
              }
          }
      }
    if (auto it = doc.FindMember("boundaries"); it != doc.MemberEnd())
      if (it->value.IsArray()) {
        fmt::print(fg(fmt::color::cyan), "{} boundaries in total\n",
                   it->value.Size());
        for (auto &boundary : it->value.GetArray()) {
          std::string fn = boundary["file"].GetString();
          fmt::print("loading boundary {}\n", fn);
          if(boundary["format"] == "vdb")
          {
            boundaries.push_back(readPhiFromVdbFile(fn));
          }
          else if(boundary["format"]=="obj")
          {
              boundaries.push_back(vdbToolsWapper::readMeshToLevelset(fn, simulator.dx));
          }
        }
      }
  }
}
void parse_scene(std::string fn, Aero &simulator) {
    fs::path p{fn};
    if (p.empty())
        fmt::print("file not exist {}\n", fn);
    else
    {
        std::size_t size = fs::file_size(p);
        std::string configs;
        configs.resize(size);

        std::ifstream istrm(fn);
        if (!istrm.is_open())
            fmt::print("cannot open file {}\n", fn);
        else
            istrm.read(const_cast<char *>(configs.data()), configs.size());
        istrm.close();
        fmt::print("load the scene file of size {}\n", size);

        rj::Document doc;
        doc.Parse(configs.data());
        /// simulation
        if (auto it = doc.FindMember("simulation"); it != doc.MemberEnd())
        {
            if (auto &sim = it->value; sim.IsObject()) {
                simConfigs.dx = sim["dx"].GetFloat();
                simConfigs.dt = sim["dt"].GetFloat();
                simConfigs.T  = sim["T"].GetFloat();
                simConfigs.g  = sim["G"].GetFloat();
                simConfigs.outputDir = sim["output_dir"].GetString();
                fmt::print(fg(fmt::color::cyan), "simulation: dx[{}], outputDir[{}]\n",
                           simConfigs.dx, simConfigs.outputDir);
                simulator.dx = simConfigs.dx;
            }
        }
        if (auto it = doc.FindMember("domains"); it != doc.MemberEnd())
        {
            if (it->value.IsArray())
            {
                fmt::print(fg(fmt::color::cyan), "{} domains in total\n",
                           it->value.Size());
                for (auto &particles : it->value.GetArray())
                {

                    std::string fn = particles["file"].GetString();
                    FluidSim* newSimulator = new FluidSim();
                    simulator.domains.push_back(newSimulator);
                    parse_scene(fn, *(simulator.getLatestDomain()));
                    simulator.getLatestDomain()->init_domain();

                }
            }
        }
        if (auto it = doc.FindMember("tracers"); it != doc.MemberEnd())
        {
            if (it->value.IsArray())
            {
                fmt::print(fg(fmt::color::cyan), "{} tracers in total\n",
                           it->value.Size());
                for (auto &tracer : it->value.GetArray())
                {
                    std::string fn = tracer["file"].GetString();
                    fmt::print("loading tracer {}\n", fn);
                    if (tracer["format"] == "vdb")
                    {
                        simulator.tracerEmitters.push_back(readPhiFromVdbFile(fn));
                    } else if (tracer["format"] == "obj")
                    {
                        simulator.tracerEmitters.push_back(vdbToolsWapper::readMeshToLevelset(fn, simulator.dx));
                    }
                }
            }
        }
        if (auto it = doc.FindMember("boundaries"); it != doc.MemberEnd())
        {
            if (it->value.IsArray())
            {
                fmt::print(fg(fmt::color::cyan), "{} boundaries in total\n",
                           it->value.Size());
                for (auto &boundary : it->value.GetArray())
                {
                    std::string fn = boundary["file"].GetString();
                    fmt::print("loading boundary {}\n", fn);
                    if (boundary["format"] == "vdb")
                    {
                        boundaries.push_back(readPhiFromVdbFile(fn));
                    } else if (boundary["format"] == "obj")
                    {
                        boundaries.push_back(vdbToolsWapper::readMeshToLevelset(fn, simulator.dx));
                    }
                }
            }
        }

    }
}




#endif
