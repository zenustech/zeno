#include "fluidsim.h"
#include "sparse_matrix.h"

#include "tbb/tbb.h"
#include "volumeMeshTools.h"

void FluidSim::initialize(double _dx) {
  dx = _dx;
  particle_radius = (float)(dx * 1.01 * sqrt(3.0) / 2.0);
  total_frame = 0;
  particles.resize(0);
}
float FluidSim::cfl() {
  float max_vel = 0;
  vector<float> max_vels;
  max_vels.resize(eulerian_fluids.n_bulks);
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {

          max_vels[index] =
              max(fabs(eulerian_fluids.fluid_bulk[index].u.data[i]),
                  max_vels[index]);
        }
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {

          max_vels[index] =
              max(fabs(eulerian_fluids.fluid_bulk[index].v.data[i]),
                  max_vels[index]);
        }
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {

          max_vels[index] =
              max(fabs(eulerian_fluids.fluid_bulk[index].w.data[i]),
                  max_vels[index]);
        }
      });
  for (uint i = 0; i < max_vels.size(); i++) {
    max_vel = max(max_vel, max_vels[i]);
  }
  cout << "max vel:" << max_vel << endl;
  float dt = dx / max_vel;

  vector<float> temp;
  max_vels.swap(temp);
  return dt;
}

void FluidSim::set_liquid(const FLUID::Vec3f &bmin, const FLUID::Vec3f &bmax,
                          std::function<float(const FLUID::Vec3f &)> phi) {
    std::cout<<"filling liquid\n";
  float gap = 0.5 * dx;
  cout << "gap between particles: " << gap << endl;
  cout << "bmin: " << bmin[0] << " " << bmin[1] << " " << bmin[2] << endl;
  cout << "bmax: " << bmax[0] << " " << bmax[1] << " " << bmax[2] << endl;
  FLUID::Vec3i nijk = FLUID::Vec3i((bmax - bmin) / gap);
  cout << "bmin: " << bmin[0] << " " << bmin[1] << " " << bmin[2] << endl;
  cout << "bmax: " << bmax[0] << " " << bmax[1] << " " << bmax[2] << endl;
  cout << "particle dimensions: " << nijk[0] << " " << nijk[1] << " " << nijk[2]
       << endl;
    int ni=nijk[0], nj=nijk[1],nk=nijk[2];
    size_t numVoxel = (size_t)ni*(size_t)nj*(size_t)nk;
    size_t current_cnt = particles.size();
    std::vector<char> buffer(numVoxel);
    buffer.assign(numVoxel,0);
    std::vector<size_t> indexBuffer(numVoxel);
    std::cout<<"buffer resize done\n";
    tbb::parallel_for((size_t)0, (size_t)ni*nj*nk, (size_t)1, [&](size_t index)
    {
        int i,j,k;
        i = index%ni; j = (index%(ni*nj))/ni; k = index/(ni*nj);
        FLUID::Vec3f pos = FLUID::Vec3f(i, j, k) * gap + bmin;
        if (phi(pos) <= 0 && pos[0] >= bmin[0] && pos[1] >= bmin[1] &&
            pos[2] >= bmin[2] && pos[0] <= bmax[0] && pos[1] <= bmax[1] &&
            pos[2] <= bmax[2])
        {
            buffer[index] = 1;
        }

    });
    std::exclusive_scan(std::execution::par, buffer.begin(), buffer.end(),
                        indexBuffer.begin(), 0);
    std::cout<<"scan done\n";
    auto cnt = indexBuffer.back() + buffer.back();
    std::cout<<cnt<<std::endl;
    particles.resize(cnt+current_cnt);
    tbb::parallel_for(
            (size_t)0, (size_t)buffer.size(), (size_t)1, [&](size_t buffer_id) {
                if (buffer[buffer_id] == 1) {
                    int ii = buffer_id % ni, jj = (buffer_id % (ni * nj)) / ni,
                            kk = buffer_id / (ni * nj);
                    int id = indexBuffer[buffer_id];
                    FLUID::Vec3f pos = FLUID::Vec3f(ii, jj, kk) * gap + bmin;
                    particles[current_cnt+id] = FLIP_particle(pos, FLUID::Vec3f(0, 0, 0));
                }
            });
//  for (int k = 0; k < nijk[2]; k++)
//    for (int j = 0; j < nijk[1]; j++)
//      for (int i = 0; i < nijk[0]; i++) {
//        FLUID::Vec3f pos = FLUID::Vec3f(i, j, k) * gap + bmin;
//        if (phi(pos) <= 0 && pos[0] >= bmin[0] && pos[1] >= bmin[1] &&
//            pos[2] >= bmin[2] && pos[0] <= bmax[0] && pos[1] <= bmax[1] &&
//            pos[2] <= bmax[2])
//          particles.push_back(FLIP_particle(pos, FLUID::Vec3f(0, 0, 0)));
//      }
}
void FluidSim::init_domain() {
  eulerian_fluids.initialize_bulks(particles, dx);
    fusion_p2g_liquid_phi(eulerian_fluids, particles, dx, particle_radius);
}
void FluidSim::emitFluids(float dt, float (*phi)(const FLUID::Vec3f &)) {
  for (int k = 0; k < emitters.size(); k++) {
    boxEmitter emitter = emitters[k];
    int slices = floor(dt * mag(emitter.vel) / dx) * 2 + 1;
    FLUID::Vec3f b = emitter.bmax - emitter.bmin;
    int w = floor(b[0] / dx * 2.0);
    int h = floor(b[1] / dx * 2.0);
    int n = particles.size();
    particles.resize(particles.size() + slices * w * h);
    int totalCnt = 0;
    for (int i = 0; i < slices; i++) {
      int cnt = 0;
      for (int jj = 0; jj < h; jj++)
        for (int ii = 0; ii < w; ii++) {
          FLUID::Vec3f pos =
              emitter.bmin +
              FLUID::Vec3f(((float)ii + 0.5) / (float)w * b[0],
                           ((float)jj + 0.5) / (float)h * b[1], 0) +
              (float)i / (float)slices * dt * emitter.vel;
          cnt++;
          particles[n + totalCnt] = FLIP_particle(pos, emitter.vel);
          totalCnt++;
        }
    }
    printf("emitted %d particles\n", totalCnt);
  }
  std::vector<FLIP_particle> newParticles;
  newParticles.resize(particles.size());
  int cnt = 0;
  for (size_t i = 0; i < particles.size(); i++) {
    FLUID::Vec3f pos = particles[i].pos;
    if (phi(pos) >= 0 && pos[0] >= regionMin[0] && pos[1] >= regionMin[1] &&
        pos[2] >= regionMin[2] && pos[0] <= regionMax[0] &&
        pos[1] <= regionMax[1] && pos[2] <= regionMax[2]) {
      newParticles[cnt++] = particles[i];
    }
  }
  newParticles.resize(cnt);
  particles.swap(newParticles);
  std::vector<FLIP_particle> temp;
  newParticles.swap(temp);
}
bool FluidSim::isIsolatedParticle(FLUID::Vec3f &pos) {
  float phi = eulerian_fluids.get_liquid_phi(pos);
  //    FLUID::Vec3f pos2 = pos - eulerian_fluids.bmin;
  //    FLUID::Vec3i ijk = FLUID::Vec3i(pos2/dx);
  //    int bulkidx=eulerian_fluids.find_bulk(ijk[0],ijk[1],ijk[2]);
  //    int cnt=0;
  //    for(int k=ijk[2]-1;k<=ijk[2]+1;k++)
  //        for(int j=ijk[1]-1;j<=ijk[1]+1;j++)
  //            for(int i=ijk[0]-1;i<=ijk[0]+1;i++)
  //            {
  //
  //                if(eulerian_fluids.liquid_phi(bulkidx,i,j,k)<0)
  //                {
  //                    cnt++;
  //                }
  //            }
  return phi > 0;
}
void FluidSim::advect_particles(float dt) {
  // particles are already sorted to maximize RAM hit rate
  tbb::parallel_for((size_t)0, (size_t)particles.size(), (size_t)1,
                    [&](size_t index) {
                      FLUID::Vec3f pos = particles[index].pos;
                        if(eulerian_fluids.get_liquid_phi(pos)<0)
                            pos += dt * eulerian_fluids.get_velocity(pos);
                        else
                            pos += dt * particles[index].vel;
                      //        if(eulerian_fluids.isValidVel(pos)&&(!eulerian_fluids.isIsolated(pos)))
                      //        {

                      // pos = trace_rk3(particles[index].pos, dt);

                      //        } else{
                      //            particles[index].vel +=
                      //            0.5f*dt*FLUID::Vec3f(0,-9.8,0); pos +=
                      //            dt*particles[index].vel;
                      //            particles[index].vel +=
                      //            0.5f*dt*FLUID::Vec3f(0,-9.8,0);
                      //        }

//                      float phi_val = eulerian_fluids.get_solid_phi(pos);
//                      if (phi_val < 0) {
//                        FLUID::Vec3f grad;
//                        grad = eulerian_fluids.get_grad_solid(pos);
//                        if (mag(grad) > 0)
//                          normalize(grad);
//                        pos -= phi_val * grad;
//                      }
//                      else if (phi_val < dx)
//                      {
//
//                      }
                      particles[index].pos = pos;
                    });
}
void FluidSim::resolveParticleBoundaryCollision() {
  // particles are already sorted to maximize RAM hit rate
  tbb::parallel_for(
      (size_t)0, (size_t)particles.size(), (size_t)1, [&](size_t index) {
        FLUID::Vec3f pos = particles[index].pos;
        // check boundaries and project exterior particles back in
        float phi_val = eulerian_fluids.get_solid_phi(pos);
        if (phi_val < 0) {
          FLUID::Vec3f grad;
          grad = eulerian_fluids.get_grad_solid(particles[index].pos);
          if (mag(grad) > 0)
            normalize(grad);
          pos -= phi_val * grad;
        }
              particles[index].pos = pos;
      });
}
void FluidSim::FLIP_advection(float dt) {
  // for all eulerian bulks, u_coef.zero, u_delta.zero;
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].u_delta.data[i] = 0;
                      }
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].v_delta.data[i] = 0;
                      }
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].w_delta.data[i] = 0;
                      }
                    });

  // compute delta,
  // for all eulerian bulks, compute u_delta
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].u_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].u.data[i] -
                            eulerian_fluids.fluid_bulk[index].u_save.data[i];
                      }
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].v_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].v.data[i] -
                            eulerian_fluids.fluid_bulk[index].v_save.data[i];
                      }
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].w_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].w.data[i] -
                            eulerian_fluids.fluid_bulk[index].w_save.data[i];
                      }
                    });

  // for each particle, p.vel = alpha*Interp(U) + (1-alpha)*(U_p + Interp(dU))
  particle_interpolate(0.03);
  // move particle
  float t = 0;
  float substep = dt;
  while (t < dt) {

    if (t + substep > dt)
      substep = dt - t;
    advect_particles(substep);
    resolveParticleBoundaryCollision();
    t += substep;
  }

  //
}
bool FluidSim::inDomain(FLUID::Vec3f pos) {
    float phi = 1.f;
    for (auto &grid : fluidDomains) {
        openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
                grid->constTree(), grid->transform());
        openvdb::math::Vec3<float> P(pos[0], pos[1], pos[2]);
        auto tmp = interpolator.wsSample(P); // ws denotes world space
        if (tmp < phi)
            phi = tmp;
        if (phi < 0.f)
            break;
    }
    if(fluidDomains.size()==0)
        return true;
    else
        return (float)phi<0.0f;

}
bool FluidSim::inFluid(FLUID::Vec3f pos) {
//    float phi = 1.f;
//    for (auto &grid : fluidDomains) {
//        openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
//                grid->constTree(), grid->transform());
//        openvdb::math::Vec3<float> P(pos[0], pos[1], pos[2]);
//        auto tmp = interpolator.wsSample(P); // ws denotes world space
//        if (tmp < phi)
//            phi = tmp;
//        if (phi < 0.f)
//            break;
//    }
//    if(fluidDomains.size()==0)
//        return true;
//    else
//        return (float)phi<0.0f;
    FLUID::Vec3i ijk = FLUID::Vec3i((pos - eulerian_fluids.bmin)/dx);
    if(eulerian_fluids.find_bulk(ijk[0],ijk[1],ijk[2])>=0)
    {
        return true;
    }
    return false;
}
float FluidSim::sampleEmitter(FLUID::Vec3f &pos, levelsetEmitter & lse) {
    float phi = 1.f;
    //for (auto &lse : Emitters) {
    auto grid = lse.ls;
    openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
            grid->constTree(), grid->transform());
    openvdb::math::Vec3<float> P(pos[0], pos[1], pos[2]);
    return (float)(interpolator.wsSample(P)); // ws denotes world space

}

void FluidSim::emitRegion(float (*phi)(const FLUID::Vec3f &), float dt) {
    if(fillRegion)
    {

        for (auto &grid : fluidDomains) {
            std::vector <std::vector<FLIP_particle>> emitParticles;
            emitParticles.resize(eulerian_fluids.n_bulks);
            tbb::parallel_for((size_t) 0, (size_t) eulerian_fluids.n_bulks, (size_t) 1, [&](size_t index) {
                FLUID::Vec3f corner = FLUID::Vec3f(eulerian_fluids.fluid_bulk[index].tile_corner)*dx + eulerian_fluids.bmin;
                if(eulerian_fluids.fluid_bulk[index].has_hole
                && (inDomain(corner) || inDomain(corner+FLUID::Vec3f(8,0,0)*dx) || inDomain(corner+FLUID::Vec3f(0,8,0)*dx) || inDomain(corner+FLUID::Vec3f(8,8,0)*dx)
                ||  inDomain(corner+FLUID::Vec3f(0,0,8)*dx)   || inDomain(corner+FLUID::Vec3f(8,0,8)*dx) || inDomain(corner+FLUID::Vec3f(0,8,8)*dx) || inDomain(corner+FLUID::Vec3f(8,8,8)*dx)) )
                for (int kk = 0; kk < 8; kk++)
                    for (int jj = 0; jj < 8; jj++)
                        for (int ii = 0; ii < 8; ii++) {
                            FLUID::Vec3i ijk = FLUID::Vec3i(eulerian_fluids.fluid_bulk[index].tile_corner + FLUID::Vec3i(ii,jj,kk));
                            FLUID::Vec3f pos = eulerian_fluids.bmin + FLUID::Vec3f(ijk[0],ijk[1],ijk[2])  * dx + FLUID::Vec3f(0.5 * dx);
                            for(int iii=-1;iii<=1;iii+=2)for(int jjj=-1;jjj<=1;jjj+=2)for(int kkk=-1;kkk<=1;kkk+=2) {

                                        FLUID::Vec3f epos = pos + 0.25f * dx * FLUID::Vec3f(iii, jjj, kkk);
                                        if (inDomain(epos)&&phi(epos)>0) {
                                            FLUID::Vec3f vel = eulerian_fluids.get_velocity(epos);
                                            if (EmitterSampleField != nullptr) {
                                                vel = EmitterSampleField->getVelocity(epos);
                                            }
                                            if (eulerian_fluids.get_liquid_phi(epos) > 0 && inDomain(epos) &&
                                                phi(epos) > 0) {
                                                emitParticles[index].emplace_back(FLIP_particle(epos, vel));
                                            }
                                            FLUID::Vec3f dpos = epos - dt * vel;
                                            FLUID::Vec3f dvel = eulerian_fluids.get_velocity(dpos);
                                            int seed_num = (dt * FLUID::mag(dvel) / dx) + 1;
                                            for (int seeded = 0; seeded <= seed_num; seeded++) {
                                                dpos += (float) seeded * dt / (float) seed_num * dvel;
                                                vel = eulerian_fluids.get_velocity(dpos);
                                                if (EmitterSampleField != nullptr) {
                                                    vel = EmitterSampleField->getVelocity(dpos);
                                                }
                                                epos = dpos + dt * vel;
                                                if (eulerian_fluids.get_liquid_phi(epos) >= 0 && inDomain(epos) &&
                                                    phi(epos) > 0) {
                                                    emitParticles[index].emplace_back(FLIP_particle(dpos, vel));
                                                }
                                            }
                                        }
                                    }
                        }
            });
            std::cout<<""<<std::endl;
            for (int i = 0; i < emitParticles.size(); i++) {
                if(emitParticles[i].size()>0)
                    particles.insert(particles.end(), emitParticles[i].begin(), emitParticles[i].end());
            }
            std::cout << "emitter region emission done\n";

//            openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
//            auto world_min = grid->indexToWorld(box.min());
//            auto world_max = grid->indexToWorld(box.max());
//            auto bmin = FLUID::Vec3f(world_min[0], world_min[1], world_min[2]);
//            auto bmax = FLUID::Vec3f(world_max[0], world_max[1], world_max[2]);
//            float gap = 0.5 * dx;
//            FLUID::Vec3i nijk = FLUID::Vec3i((bmax - bmin) / gap);
//            int ni = nijk[0], nj=nijk[1], nk=nijk[2];
//            tbb::parallel_for((size_t)0, (size_t)ni*nj*nk, (size_t)1, [&](size_t index)
//            {
//                int i = index%ni;
//                int j = (index/ni)%nj;
//                int k = index/(ni*nj);
//                auto pos = bmin + gap * FLUID::Vec3f(i,j,k);
//                FLUID::Vec3f vel = eulerian_fluids.get_velocity(pos);
//                if(EmitterSampleField!= nullptr)
//                {
//                    vel = EmitterSampleField->getVelocity(pos);
//                }
//                if(eulerian_fluids.get_liquid_phi(pos)>=0&&inFluid(pos))
//                {
//                    emitParticles.push_back(FLIP_particle(pos, vel));
//                }
//            });
//        }
//        size_t cnt = particles.size();
//        particles.resize(emitParticles.size()+cnt);
//        tbb::parallel_for((size_t)0, (size_t)emitParticles.size(), (size_t)1, [&](size_t index)
//        {
//            particles[cnt+index] = emitParticles[index];
//        });
        }
    }

}
void FluidSim::emit(std::vector<FLIP_particle>& p, levelsetEmitter &lse, sparse_fluid8x8x8 &_eulerian_fluids, FluidSim* _emitter_sample_field, float _gap, float (*phi)(const FLUID::Vec3f &))
{

    auto grid = lse.ls;
    sparse_fluid8x8x8 emitter_fluids;
    std::vector<std::vector<FLIP_particle>> emitParticles(0);
//    struct emitBody{
//        using GridT = typename openvdb::Grid<typename openvdb::tree::Tree4<float, 5, 4, 3>::Type>;
//        using TreeT = typename GridT::TreeType;
//        using IterRange = openvdb::tree::IteratorRange<typename TreeT::LeafCIter>;
//        std::vector<std::vector<FLIP_particle>> &mEmitParticles(0);
//        levelsetEmitter &mlse;
//        FluidSim* mEmitterSampleFiled;
//        sparse_fluid8x8x8 &mEulerianFluids;
//        emitBody(std::vector<std::vector<FLIP_particle>> &_ep, levelsetEmitter &_lse, FluidSim* _esf, sparse_fluid8x8x8 &_eulerian_fluids){
//            mEmitterSampleFiled = _ep;
//            mlse = _lse;
//            mEmitterSampleFiled = _esf;
//            mEulerianFluids = _eulerian_fluids;
//        }
//        void operator()(IterRange& range) const
//        {
//            // Note: this code must be thread-safe.
//            // Iterate over a subrange of the leaf iterator's iteration space.
//            for (auto leafIter = range.begin(); leafIter; ++leafIter) {
//
//                const openvdb::points::AttributeArray& array =
//                        leafIter->constAttributeArray("P");
//                openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);
//                for(auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
//                    openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
//                    FLUID::Vec3f vel = mlse.vel;
//                    if(mEmitterSampleFiled!= nullptr)
//                    {
//                        vel = mEmitterSampleFiled->getVelocity(FLUID::Vec3f(voxelPosition[0],voxelPosition[1],voxelPosition[2]));
//                    }
//                    if(mEulerianFluids.get_liquid_phi(pos)>=0)
//                    {
//                        mEmitParticles[index].push_back(FLIP_particle(pos, vel));
//                    }
//                }
//            }
//        }
//    };
//    auto emitBody vdbEmitter(emitParticles, lse, _emitter_sample_field, _eulerian_fluids);
//    emitBody::IterRange range(lse.ls->tree().cbeginLeaf());
//    tbb::parallel_for(range, emitBody);



    emitter_fluids.initialize_bulks(grid, _gap);
    emitParticles.resize(emitter_fluids.n_bulks);
    for(int i=0;i<emitParticles.size();i++)
    {
        emitParticles[i].reserve(512);
    }
    std::cout<<"emitter vel:"<<lse.vel<<std::endl;
    tbb::parallel_for((size_t)0, (size_t)emitter_fluids.n_bulks, (size_t)1, [&](size_t index)
    {
        for(int kk=0;kk<8;kk++)for(int jj=0;jj<8;jj++)for(int ii=0;ii<8;ii++) {
                    FLUID::Vec3i ijk = FLUID::Vec3i(emitter_fluids.fluid_bulk[index].tile_corner + FLUID::Vec3i(ii,jj,kk));
                    FLUID::Vec3f pos = emitter_fluids.bmin + FLUID::Vec3f(ijk[0],ijk[1],ijk[2])  * _gap + FLUID::Vec3f(0.5 * _gap);
                    FLUID::Vec3f vel = lse.vel;
                    if(_emitter_sample_field!= nullptr)
                    {
                        vel = _emitter_sample_field->getVelocity(pos);
                    }
                    if(_eulerian_fluids.get_liquid_phi(pos)>=0&&sampleEmitter(pos, lse)<=0&&phi(pos)>0)
                    {
                        emitParticles[index].push_back(FLIP_particle(pos, vel));
                    }
                }
    });
//    for(int i=0;i<emitParticles.size();i++)
//    {
//        std::cout<<emitParticles[i].size()<<std::endl;
//    }
    tbb::parallel_for((size_t)0, (size_t)_eulerian_fluids.n_bulks, (size_t)1, [&](size_t index)
    {
        for(int ii=0;ii<8;ii++)for(int jj=0;jj<8;jj++)for(int kk=0;kk<8;kk++) {
                    FLUID::Vec3i ijk = FLUID::Vec3i(_eulerian_fluids.fluid_bulk[index].tile_corner + FLUID::Vec3i(ii,jj,kk));
                    FLUID::Vec3f pos = _eulerian_fluids.bmin + FLUID::Vec3f(ijk[0],ijk[1],ijk[2])  * _eulerian_fluids.h + FLUID::Vec3f(0.5 * _eulerian_fluids.h);
                    _eulerian_fluids.liquid_phi(index, ii,jj,kk) = std::min(sampleEmitter(pos, lse), _eulerian_fluids.liquid_phi(index, ii,jj,kk));
                }
    });
    tbb::parallel_for((size_t)0, (size_t)p.size(), (size_t)1, [&](size_t index) {
        if (_emitter_sample_field == nullptr && sampleEmitter(p[index].pos, lse) < 0 ) {
            p[index].vel = lse.vel;
        }
    });
    for(int i=0;i<emitParticles.size();i++)
    {
        p.insert(p.end(), emitParticles[i].begin(), emitParticles[i].end());
    }
    std::cout<<"emitter emission done\n";












}
void FluidSim::emit(float (*phi)(const FLUID::Vec3f &))
{
//    tbb::concurrent_vector<FLIP_particle> emitParticles;
//    emitParticles.reserve(particles.size());
//    emitParticles.resize(0);
//    for (auto &lse : Emitters) {
//        auto grid = lse.ls;
//        openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
//        auto world_min = grid->indexToWorld(box.min());
//        auto world_max = grid->indexToWorld(box.max());
//        auto bmin = FLUID::Vec3f(world_min[0], world_min[1], world_min[2]);
//        auto bmax = FLUID::Vec3f(world_max[0], world_max[1], world_max[2]);
//        float gap = 0.5 * dx;
//        FLUID::Vec3i nijk = FLUID::Vec3i((bmax - bmin) / gap);
//        int ni = nijk[0], nj=nijk[1], nk=nijk[2];
//        tbb::parallel_for((size_t)0, (size_t)ni*nj*nk, (size_t)1, [&](size_t index)
//        {
//            int i = index%ni;
//            int j = (index/ni)%nj;
//            int k = index/(ni*nj);
//            auto pos = bmin + gap * FLUID::Vec3f(i,j,k);
//            FLUID::Vec3f vel = lse.vel;
//            if(EmitterSampleField!= nullptr)
//            {
//                vel = EmitterSampleField->getVelocity(pos);
//            }
//            if(eulerian_fluids.get_liquid_phi(pos)>=0&&sampleEmitter(pos, lse)<0)
//            {
//                emitParticles.push_back(FLIP_particle(pos, vel));
//            }
//        });
//        tbb::parallel_for((size_t)0, (size_t)particles.size(), (size_t)1, [&](size_t index)
//        {
//            if(EmitterSampleField == nullptr && sampleEmitter(particles[index].pos,lse)<0)
//            {
//                particles[index].vel = lse.vel;
//            }
//        });
//    }
//    size_t cnt = particles.size();
//    particles.resize(emitParticles.size()+cnt);
//    tbb::parallel_for((size_t)0, (size_t)emitParticles.size(), (size_t)1, [&](size_t index)
//    {
//        particles[cnt+index] = emitParticles[index];
//    });
    for (auto &lse:Emitters)
    {
        emit(particles, lse, eulerian_fluids, EmitterSampleField, 0.5*dx, phi);
//        auto grid = lse.ls;
//        sparse_fluid8x8x8 emitter_fluids;
//        std::vector<std::vector<FLIP_particle>> emitParticles;
//        emitter_fluids.initialize_bulks(grid, 0.5*dx);
//        emitParticles.resize(emitter_fluids.n_bulks);
//
////        openvdb::points::PointDataTree::Ptr pointTree(
////                new openvdb::points::PointDataTree(grid->tree(), 0, openvdb::TopologyCopy()));
////        pointTree->voxelizeActiveTiles();
////
//        tbb::parallel_for((size_t)0, (size_t)emitter_fluids.n_bulks, (size_t)1, [&](size_t index)
//        {
//            for(int ii=0;ii<8;ii++)for(int jj=0;jj<8;jj++)for(int kk=0;kk<8;kk++) {
//                        FLUID::Vec3f pos =
//                                emitter_fluids.bmin + FLUID::Vec3f(emitter_fluids.fluid_bulk[index].tile_corner+FLUID::Vec3i(ii,jj,kk))  * 0.5 * dx +
//                                FLUID::Vec3f(0.25 * dx);
//                        FLUID::Vec3f vel = lse.vel;
//                        if(EmitterSampleField!= nullptr)
//                        {
//                            vel = EmitterSampleField->getVelocity(pos);
//                        }
//                        if(eulerian_fluids.get_liquid_phi(pos)>=0&&sampleEmitter(pos, lse)<0)
//                        {
//                            emitParticles[index].emplace_back(FLIP_particle(pos, vel));
//                        }
//                    }
//        });
//        tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index)
//        {
//            for(int ii=0;ii<8;ii++)for(int jj=0;jj<8;jj++)for(int kk=0;kk<8;kk++) {
//                        FLUID::Vec3f pos =
//                                eulerian_fluids.bmin + FLUID::Vec3f(eulerian_fluids.fluid_bulk[index].tile_corner+FLUID::Vec3i(ii,jj,kk)) * dx +
//                                FLUID::Vec3f(0.5 * dx);
//                        eulerian_fluids.liquid_phi(index, ii,jj,kk) = std::min(sampleEmitter(pos, lse), eulerian_fluids.liquid_phi(index, ii,jj,kk));
//                    }
//        });
//        tbb::parallel_for((size_t)0, (size_t)particles.size(), (size_t)1, [&](size_t index) {
//            if (EmitterSampleField == nullptr && sampleEmitter(particles[index].pos, lse) < 0) {
//                particles[index].vel = lse.vel;
//            }
//        });
//        for(int i=0;i<emitParticles.size();i++)
//        {
//            particles.insert(particles.end(), emitParticles[i].begin(), emitParticles[i].end());
//        }
//        std::cout<<"emitter emission done\n";
    }


}
void FluidSim::remeshing() {

std::cout<<"remeshing..."<<std::endl;
    std::vector<char> cnt(particles.size());
    cnt.assign(particles.size(),1);
    tbb::parallel_for((size_t)0, (size_t)particles.size(), (size_t)1, [&](size_t index){
        if(eulerian_fluids.get_solid_phi(particles[index].pos)<0)
        {
            cnt[index] = 0;
        }
    });
    auto numvoxel = eulerian_fluids.fluid_bulk.size()*eulerian_fluids.n_perbulk;
    unsigned char *pinc;
    pinc = new unsigned char[numvoxel];
    memset(pinc, 0, numvoxel);
#pragma omp parallel for
    for (int index = 0; index < particles.size(); index++) {
        FLUID::Vec3f pos = particles[index].pos;
        FLUID::Vec3i ijk = FLUID::Vec3i((pos-eulerian_fluids.bmin)/dx);
        auto bulkid = eulerian_fluids.find_bulk(ijk[0],ijk[1],ijk[2]);
        if(bulkid>=0&&bulkid<eulerian_fluids.fluid_bulk.size()) {
            FLUID::Vec3i ijk2 = ijk - eulerian_fluids.fluid_bulk[bulkid].tile_corner;
            int ii = ijk2[0] + ijk2[1] * 8 + ijk2[2] * 64;
            if(ii>=0&&ii<512) {
                size_t mem_idx = bulkid * eulerian_fluids.n_perbulk + ii;
                unsigned char a;
                if(mem_idx<numvoxel) {
#pragma omp atomic capture
                    a = pinc[mem_idx]++;
                    if (a > 12) {
                        cnt[index] = 0;
                    }
                }
            }
        }
    }
    std::cout<<"count ppc done\n";
//    if(fluidDomains.size()>0) {
//
//	    tbb::parallel_for((size_t)0, (size_t)particles.size(), (size_t)1, [&](size_t index){
//		    if(!inFluid(particles[index].pos))
//		    {
//		        cnt[index] = 0;
//		    }
//		});
//    }
    subset_particles(particles, particles, cnt);
    delete pinc;
    std::cout<<"remeshing done\n";

}

void FluidSim::subset_particles(std::vector<FLIP_particle> &pin,std::vector<FLIP_particle> &pout, std::vector<char>& mask)
{


    std::vector<int> indexBuffer(pin.size());
    indexBuffer.assign(indexBuffer.size(),0);
    std::exclusive_scan(std::execution::par, mask.begin(), mask.end(), indexBuffer.begin(), 0);
    auto total = indexBuffer.back() + mask.back();
    std::vector<FLIP_particle> temp(total);
    tbb::parallel_for((size_t)0, (size_t)mask.size(), (size_t)1, [&](size_t index){
	    if(mask[index]==1){
            temp[indexBuffer[index]] = pin[index];
        }
    });
    pout.resize(total);
    tbb::parallel_for((size_t)0, (size_t)temp.size(), (size_t)1, [&](size_t index){
        pout[index] = temp[index];
    });
}
void FluidSim::reorder_particles() {
  // try atomic
  //    float *res =
  //    vector<vector<FLIP_particle>> particle_reorder;
  //    particle_reorder.resize(eulerian_fluids.n_bulks);
  //    for (uint i=0;i<particle_reorder.size();i++)
  //    {
  //        particle_reorder[i].resize(0);
  //    }
  //
  //    for (uint i=0;i<particles.size();i++)
  //    {
  //        FLUID::Vec3f pos = particles[i].pos - eulerian_fluids.bmin;
  //        FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos/eulerian_fluids.bulk_size);
  //        uint64 idx =
  //        eulerian_fluids.get_bulk_index(bulk_ijk[0],bulk_ijk[1],bulk_ijk[2]);
  //        particle_reorder[idx].push_back(particles[i]);
  //    }
  //    particles.resize(0);
  //    particles.shrink_to_fit();
  //    for (uint i=0;i<particle_reorder.size();i++)
  //    {
  //        for (uint ii=0;ii<particle_reorder[i].size();ii++)
  //        {
  //            particles.push_back(particle_reorder[i][ii]);
  //        }
  //    }
  //    particle_reorder=vector<vector<FLIP_particle>>();
  //
  //    particle_bulks.resize(0);
  //    particle_bulks.resize(eulerian_fluids.n_bulks);
  //    vector<vector<int64>> particle_bulk_idx;
  //    particle_bulk_idx.resize(particles.size());
  //
  //    for (uint i=0;i<particles.size();i++)
  //
  //    {
  //        particle_bulk_idx[i].reserve(8);
  //    };
  //
  //    tbb::parallel_for((size_t)0, (size_t)particle_bulk_idx.size(),
  //                      (size_t)1, [&](size_t p)
  //                      {
  //                          FLUID::Vec3f pos =
  //                          particles[p].pos-eulerian_fluids.bmin; int i =
  //                          floor(pos[0]/dx); int j = floor(pos[1]/dx); int k
  //                          = floor(pos[2]/dx); std::unordered_map<int64,
  //                          int64> bulks_this_particle_is_assigned_to;
  //                          bulks_this_particle_is_assigned_to.clear();
  //                          int64 n=0;
  //                          //a particle's index can be assigned to different
  //                          bulks
  //                          //but can only be assigned to one bulk once.
  //                          for (int kk=k-1;kk<=k+1;kk++)
  //                              for (int jj=j-1;jj<=j+1;jj++)
  //                                  for (int ii=i-1;ii<=i+1;ii++)
  //                                  {
  //                                      int64 bulk_index =
  //                                      eulerian_fluids.find_bulk(ii,jj,kk);
  //                                      if
  //                                      (bulks_this_particle_is_assigned_to.find(bulk_index)==bulks_this_particle_is_assigned_to.end())
  //                                      {
  //                                          //particle_bulks[bulk_index].push_back(p);
  //                                          particle_bulk_idx[p].push_back(bulk_index);
  //                                          bulks_this_particle_is_assigned_to[bulk_index]
  //                                          = n; n++;
  //                                      }
  //                                  }
  //                          bulks_this_particle_is_assigned_to.clear();
  //                      });
  //
  //    for (size_t i=0;i<particles.size();i++)
  //    {
  //        for (int ii=0;ii<particle_bulk_idx[i].size();ii++)
  //        {
  //            particle_bulks[particle_bulk_idx[i][ii]].push_back(i);
  //        }
  //
  //    }
  //
  //    particle_bulk_idx.resize(0);
  //    particle_bulk_idx.shrink_to_fit();
}

void FluidSim::find_surface_particles(
    std::vector<FLIP_particle> &outParticles) {
  particle_to_grid_mask();
  outParticles.reserve(particles.size());
  outParticles.clear();
  for (size_t i = 0; i < particles.size(); i++) {
    FLUID::Vec3f pos = particles[i].pos - (eulerian_fluids.bmin +
                                           dx * FLUID::Vec3f(0.5, 0.5, 0.5));
    FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * eulerian_fluids.h));
    FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / eulerian_fluids.h);

    int bulkidx = eulerian_fluids.find_bulk(particle_ijk[0], particle_ijk[1],
                                            particle_ijk[2]);
    if (eulerian_fluids.fluid_bulk[bulkidx].mask == 1)
      outParticles.push_back(particles[i]);
  }
}
void FluidSim::particle_to_grid_mask() {
  // particle to grid
  // for all eulerian bulk u.zero;
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      eulerian_fluids.fluid_bulk[index].mark = 0;
                      eulerian_fluids.fluid_bulk[index].mask = 0;
                    });

  tbb::parallel_for(
      (size_t)0, (size_t)particles.size(), (size_t)1, [&](size_t index) {
        FLUID::Vec3f pos =
            particles[index].pos -
            (eulerian_fluids.bmin + dx * FLUID::Vec3f(0.5, 0.5, 0.5));
        FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * eulerian_fluids.h));
        FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / eulerian_fluids.h);

        int bulkidx = eulerian_fluids.find_bulk(
            particle_ijk[0], particle_ijk[1], particle_ijk[2]);
        eulerian_fluids.fluid_bulk[bulkidx].mark = 1;
      });
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        auto &bulk = eulerian_fluids.fluid_bulk[index];
        if (bulk.mark == 1) {
          auto tileCorner = bulk.tile_corner;
          for (int dx = -8; dx <= 8; dx += 8)
            for (int dy = -8; dy <= 8; dy += 8)
              for (int dz = -8; dz <= 8; dz += 8) {
                auto corner = tileCorner + FLUID::Vec3i{dx, dy, dz};
                int bulkidx =
                    eulerian_fluids.find_bulk(corner[0], corner[1], corner[2]);
                if (eulerian_fluids.fluid_bulk[bulkidx].mark == 0)
                  bulk.mask = 1;
              }
        }
      });
}
void FluidSim::particle_to_grid() {
  // particle to grid
  // for all eulerian bulk u.zero;
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].u.data[i] = 0;
                        eulerian_fluids.fluid_bulk[index].v.data[i] = 0;
                        eulerian_fluids.fluid_bulk[index].w.data[i] = 0;
                        eulerian_fluids.fluid_bulk[index].u_coef.data[i] = 1e-4;
                        eulerian_fluids.fluid_bulk[index].v_coef.data[i] = 1e-4;
                        eulerian_fluids.fluid_bulk[index].w_coef.data[i] = 1e-4;
                      }
                    });
  // try atomic
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].global_index.data[i] =
                            index * 512 + i;
                      }
                    });

  int n = eulerian_fluids.n_bulks * eulerian_fluids.n_perbulk;
  float *resu = new float[n];
  float *resv = new float[n];
  float *resw = new float[n];
  float *reswu = new float[n];
  float *reswv = new float[n];
  float *resww = new float[n];
  memset(resu, 0, sizeof(float) * n);
  memset(resv, 0, sizeof(float) * n);
  memset(resw, 0, sizeof(float) * n);
  memset(reswu, 0, sizeof(float) * n);
  memset(reswv, 0, sizeof(float) * n);
  memset(resww, 0, sizeof(float) * n);

#pragma omp parallel for
  for (int i = 0; i < particles.size(); i++) {
    FLUID::Vec3f pos = particles[i].pos - (eulerian_fluids.bmin +
                                           dx * FLUID::Vec3f(0.0, 0.5, 0.5));
    FLUID::Vec3f vel = particles[i].vel;
    FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * eulerian_fluids.h));
    FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / eulerian_fluids.h);
    FLUID::Vec3i particle_local_ijk = particle_ijk - 8 * bulk_ijk;
    for (int kk = particle_local_ijk[2] - 1; kk <= particle_local_ijk[2] + 1;
         kk++)
      for (int jj = particle_local_ijk[1] - 1; jj <= particle_local_ijk[1] + 1;
           jj++)
        for (int ii = particle_local_ijk[0] - 1;
             ii <= particle_local_ijk[0] + 1; ii++) {
          int bulkidx = eulerian_fluids.find_bulk(
              particle_ijk[0], particle_ijk[1], particle_ijk[2]);
          int mem_idx = eulerian_fluids.global_index(bulkidx, ii, jj, kk);

          FLUID::Vec3f sample_pos_u =
              dx * FLUID::Vec3f(8 * bulk_ijk + FLUID::Vec3i(ii, jj, kk));

          float weight0 = FluidSim::compute_weight(sample_pos_u, pos, dx);
#pragma omp atomic
          resu[mem_idx] += (float)(weight0 * vel[0]);
#pragma omp atomic
          reswu[mem_idx] += (float)weight0;
        }
  }

#pragma omp parallel for
  for (int i = 0; i < particles.size(); i++) {
    FLUID::Vec3f pos = particles[i].pos - (eulerian_fluids.bmin +
                                           dx * FLUID::Vec3f(0.5, 0.0, 0.5));
    FLUID::Vec3f vel = particles[i].vel;
    FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * eulerian_fluids.h));
    FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / eulerian_fluids.h);
    FLUID::Vec3i particle_local_ijk = particle_ijk - 8 * bulk_ijk;
    for (int kk = particle_local_ijk[2] - 1; kk <= particle_local_ijk[2] + 1;
         kk++)
      for (int jj = particle_local_ijk[1] - 1; jj <= particle_local_ijk[1] + 1;
           jj++)
        for (int ii = particle_local_ijk[0] - 1;
             ii <= particle_local_ijk[0] + 1; ii++) {
          int bulkidx = eulerian_fluids.find_bulk(
              particle_ijk[0], particle_ijk[1], particle_ijk[2]);
          int mem_idx = eulerian_fluids.global_index(bulkidx, ii, jj, kk);

          FLUID::Vec3f sample_pos_v =
              dx * FLUID::Vec3f(8 * bulk_ijk + FLUID::Vec3i(ii, jj, kk));

          float weight1 = FluidSim::compute_weight(sample_pos_v, pos, dx);
#pragma omp atomic
          resv[mem_idx] += (float)(weight1 * vel[1]);
#pragma omp atomic
          reswv[mem_idx] += (float)weight1;
        }
  }

#pragma omp parallel for
  for (int i = 0; i < particles.size(); i++) {
    FLUID::Vec3f pos = particles[i].pos - (eulerian_fluids.bmin +
                                           dx * FLUID::Vec3f(0.5, 0.5, 0.0));
    FLUID::Vec3f vel = particles[i].vel;
    FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * eulerian_fluids.h));
    FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / eulerian_fluids.h);
    FLUID::Vec3i particle_local_ijk = particle_ijk - 8 * bulk_ijk;
    for (int kk = particle_local_ijk[2] - 1; kk <= particle_local_ijk[2] + 1;
         kk++)
      for (int jj = particle_local_ijk[1] - 1; jj <= particle_local_ijk[1] + 1;
           jj++)
        for (int ii = particle_local_ijk[0] - 1;
             ii <= particle_local_ijk[0] + 1; ii++) {
          int bulkidx = eulerian_fluids.find_bulk(
              particle_ijk[0], particle_ijk[1], particle_ijk[2]);
          int mem_idx = eulerian_fluids.global_index(bulkidx, ii, jj, kk);

          FLUID::Vec3f sample_pos_w =
              dx * FLUID::Vec3f(8 * bulk_ijk + FLUID::Vec3i(ii, jj, kk));

          float weight2 = FluidSim::compute_weight(sample_pos_w, pos, dx);
#pragma omp atomic
          resw[mem_idx] += (float)(weight2 * vel[2]);
#pragma omp atomic
          resww[mem_idx] += (float)weight2;
        }
  }

  // spread particle velocity to grid
  //	tbb::parallel_for((size_t)0,
  //		(size_t)particle_bulks.size(),
  //		(size_t)1,
  //		[&](size_t index)
  //	{
  //
  //		for (uint i=0;i<particle_bulks[index].size();i++)
  //		{
  //			FLUID::Vec3f pos =
  // particles[particle_bulks[index][i]].pos
  //				- eulerian_fluids.bmin;
  //			FLUID::Vec3f vel =
  // particles[particle_bulks[index][i]].vel; 			FLUID::Vec3i
  // bulk_ijk
  // =
  // eulerian_fluids.fluid_bulk[index].tile_corner;
  // FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos/eulerian_fluids.h);
  // FLUID::Vec3i particle_local_ijk =
  // particle_ijk - bulk_ijk; 			for (int
  // kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
  // for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++) for
  // (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
  //			{
  //				if (kk>=0&&kk<eulerian_fluids.tile_n
  //					&&jj>=0&&jj<eulerian_fluids.tile_n
  //					&&ii>=0&&ii<eulerian_fluids.tile_n)
  //				{
  //					FLUID::Vec3f sample_pos_u
  //						=
  // dx*FLUID::Vec3f(bulk_ijk+FLUID::Vec3i(ii,jj,kk))+dx*FLUID::Vec3f(0.0,0.5,0.5);
  //					FLUID::Vec3f sample_pos_v
  //						=
  // dx*FLUID::Vec3f(bulk_ijk+FLUID::Vec3i(ii,jj,kk))+dx*FLUID::Vec3f(0.5,0.0,0.5);
  //					FLUID::Vec3f sample_pos_w
  //						=
  // dx*FLUID::Vec3f(bulk_ijk+FLUID::Vec3i(ii,jj,kk))+dx*FLUID::Vec3f(0.5,0.5,0.0);
  //
  //					float weight0 =
  // FluidSim::compute_weight(sample_pos_u,pos,dx);
  // float weight1
  // =
  // FluidSim::compute_weight(sample_pos_v,pos,dx);
  // float weight2 = FluidSim::compute_weight(sample_pos_w,pos,dx);
  //					eulerian_fluids.u(index,ii,jj,kk) +=
  // weight0*vel[0];
  // eulerian_fluids.u_coef(index,ii,jj,kk)
  // += weight0;
  //
  //					eulerian_fluids.v(index,ii,jj,kk) +=
  // weight1*vel[1];
  // eulerian_fluids.v_coef(index,ii,jj,kk)
  // += weight1;
  //
  //					eulerian_fluids.w(index,ii,jj,kk) +=
  // weight2*vel[2];
  // eulerian_fluids.w_coef(index,ii,jj,kk)
  // += weight2;
  //				}
  //			}
  //		}
  //
  //	});
  //
  //	//divide_weight(u,u_coef);
  //	//divide_weight(v,v_coef);
  //	//divide_weight(w,w_coef);
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          int gidx = eulerian_fluids.fluid_bulk[index].global_index.data[i];
          if (reswu[gidx] > 0) {
            eulerian_fluids.fluid_bulk[index].u.data[i] =
                resu[gidx] / reswu[gidx];
            eulerian_fluids.fluid_bulk[index].u_valid.data[i] = 1;
          } else {
            eulerian_fluids.fluid_bulk[index].u.data[i] = 0;
            eulerian_fluids.fluid_bulk[index].u_valid.data[i] = 0;
          }
          if (reswv[gidx] > 0) {
            eulerian_fluids.fluid_bulk[index].v.data[i] =
                resv[gidx] / reswv[gidx];
            eulerian_fluids.fluid_bulk[index].v_valid.data[i] = 1;
          } else {
            eulerian_fluids.fluid_bulk[index].v.data[i] = 0;
            eulerian_fluids.fluid_bulk[index].v_valid.data[i] = 0;
          }
          if (resww[gidx] > 0) {
            eulerian_fluids.fluid_bulk[index].w.data[i] =
                resw[gidx] / resww[gidx];
            eulerian_fluids.fluid_bulk[index].w_valid.data[i] = 1;
          } else {
            eulerian_fluids.fluid_bulk[index].w.data[i] = 0;
            eulerian_fluids.fluid_bulk[index].w_valid.data[i] = 0;
          }
        }
      });

  delete[] resu;
  delete[] resv;
  delete[] resw;
  delete[] reswu;
  delete[] reswv;
  delete[] resww;
}
void FluidSim::particle_to_grid(sparse_fluid8x8x8 &_eulerian_fluid,
                                std::vector<FLIP_particle> &_particles,
                                float _dx) {
  tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                        _eulerian_fluid.fluid_bulk[index].u.data[i] = 0;
                        _eulerian_fluid.fluid_bulk[index].v.data[i] = 0;
                        _eulerian_fluid.fluid_bulk[index].w.data[i] = 0;
                        _eulerian_fluid.fluid_bulk[index].u_coef.data[i] = 1e-4;
                        _eulerian_fluid.fluid_bulk[index].v_coef.data[i] = 1e-4;
                        _eulerian_fluid.fluid_bulk[index].w_coef.data[i] = 1e-4;
                      }
                    });
  // try atomic
  tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                        _eulerian_fluid.fluid_bulk[index].global_index.data[i] =
                            index * 512 + i;
                      }
                    });

  int n = _eulerian_fluid.n_bulks * _eulerian_fluid.n_perbulk;
  float *resu = new float[n];
  float *resv = new float[n];
  float *resw = new float[n];
  float *reswu = new float[n];
  float *reswv = new float[n];
  float *resww = new float[n];
  memset(resu, 0, sizeof(float) * n);
  memset(resv, 0, sizeof(float) * n);
  memset(resw, 0, sizeof(float) * n);
  memset(reswu, 0, sizeof(float) * n);
  memset(reswv, 0, sizeof(float) * n);
  memset(resww, 0, sizeof(float) * n);

#pragma omp parallel for
  for (int i = 0; i < _particles.size(); i++) {
    FLUID::Vec3f pos = _particles[i].pos - (_eulerian_fluid.bmin +
                                            _dx * FLUID::Vec3f(0.0, 0.5, 0.5));
    FLUID::Vec3f vel = _particles[i].vel;
    FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * _eulerian_fluid.h));
    FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / _eulerian_fluid.h);
    FLUID::Vec3i particle_local_ijk = particle_ijk - 8 * bulk_ijk;
    for (int kk = particle_local_ijk[2] - 1; kk <= particle_local_ijk[2] + 1;
         kk++)
      for (int jj = particle_local_ijk[1] - 1; jj <= particle_local_ijk[1] + 1;
           jj++)
        for (int ii = particle_local_ijk[0] - 1;
             ii <= particle_local_ijk[0] + 1; ii++) {
          int bulkidx = _eulerian_fluid.find_bulk(
              particle_ijk[0], particle_ijk[1], particle_ijk[2]);
          int mem_idx = _eulerian_fluid.global_index(bulkidx, ii, jj, kk);

          FLUID::Vec3f sample_pos_u =
              _dx * FLUID::Vec3f(8 * bulk_ijk + FLUID::Vec3i(ii, jj, kk));

          float weight0 = FluidSim::compute_weight(sample_pos_u, pos, _dx);
#pragma omp atomic
          resu[mem_idx] += (float)(weight0 * vel[0]);
#pragma omp atomic
          reswu[mem_idx] += (float)weight0;
        }
  }

#pragma omp parallel for
  for (int i = 0; i < _particles.size(); i++) {
    FLUID::Vec3f pos = _particles[i].pos - (_eulerian_fluid.bmin +
                                            _dx * FLUID::Vec3f(0.5, 0.0, 0.5));
    FLUID::Vec3f vel = _particles[i].vel;
    FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * _eulerian_fluid.h));
    FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / _eulerian_fluid.h);
    FLUID::Vec3i particle_local_ijk = particle_ijk - 8 * bulk_ijk;
    for (int kk = particle_local_ijk[2] - 1; kk <= particle_local_ijk[2] + 1;
         kk++)
      for (int jj = particle_local_ijk[1] - 1; jj <= particle_local_ijk[1] + 1;
           jj++)
        for (int ii = particle_local_ijk[0] - 1;
             ii <= particle_local_ijk[0] + 1; ii++) {
          int bulkidx = _eulerian_fluid.find_bulk(
              particle_ijk[0], particle_ijk[1], particle_ijk[2]);
          int mem_idx = _eulerian_fluid.global_index(bulkidx, ii, jj, kk);

          FLUID::Vec3f sample_pos_v =
              _dx * FLUID::Vec3f(8 * bulk_ijk + FLUID::Vec3i(ii, jj, kk));

          float weight1 = FluidSim::compute_weight(sample_pos_v, pos, _dx);
#pragma omp atomic
          resv[mem_idx] += (float)(weight1 * vel[1]);
#pragma omp atomic
          reswv[mem_idx] += (float)weight1;
        }
  }

#pragma omp parallel for
  for (int i = 0; i < _particles.size(); i++) {
    FLUID::Vec3f pos = _particles[i].pos - (_eulerian_fluid.bmin +
                                            _dx * FLUID::Vec3f(0.5, 0.5, 0.0));
    FLUID::Vec3f vel = _particles[i].vel;
    FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * _eulerian_fluid.h));
    FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / _eulerian_fluid.h);
    FLUID::Vec3i particle_local_ijk = particle_ijk - 8 * bulk_ijk;
    for (int kk = particle_local_ijk[2] - 1; kk <= particle_local_ijk[2] + 1;
         kk++)
      for (int jj = particle_local_ijk[1] - 1; jj <= particle_local_ijk[1] + 1;
           jj++)
        for (int ii = particle_local_ijk[0] - 1;
             ii <= particle_local_ijk[0] + 1; ii++) {
          int bulkidx = _eulerian_fluid.find_bulk(
              particle_ijk[0], particle_ijk[1], particle_ijk[2]);
          int mem_idx = _eulerian_fluid.global_index(bulkidx, ii, jj, kk);

          FLUID::Vec3f sample_pos_w =
              _dx * FLUID::Vec3f(8 * bulk_ijk + FLUID::Vec3i(ii, jj, kk));

          float weight2 = FluidSim::compute_weight(sample_pos_w, pos, _dx);
#pragma omp atomic
          resw[mem_idx] += (float)(weight2 * vel[2]);
#pragma omp atomic
          resww[mem_idx] += (float)weight2;
        }
  }

  tbb::parallel_for(
      (size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
          int gidx = _eulerian_fluid.fluid_bulk[index].global_index.data[i];
          if (reswu[gidx] > 0) {
            _eulerian_fluid.fluid_bulk[index].u.data[i] =
                resu[gidx] / reswu[gidx];
            _eulerian_fluid.fluid_bulk[index].u_valid.data[i] = 1;
          } else {
            _eulerian_fluid.fluid_bulk[index].u.data[i] = 0;
            _eulerian_fluid.fluid_bulk[index].u_valid.data[i] = 0;
          }
          if (reswv[gidx] > 0) {
            _eulerian_fluid.fluid_bulk[index].v.data[i] =
                resv[gidx] / reswv[gidx];
            _eulerian_fluid.fluid_bulk[index].v_valid.data[i] = 1;
          } else {
            _eulerian_fluid.fluid_bulk[index].v.data[i] = 0;
            _eulerian_fluid.fluid_bulk[index].v_valid.data[i] = 0;
          }
          if (resww[gidx] > 0) {
            _eulerian_fluid.fluid_bulk[index].w.data[i] =
                resw[gidx] / resww[gidx];
            _eulerian_fluid.fluid_bulk[index].w_valid.data[i] = 1;
          } else {
            _eulerian_fluid.fluid_bulk[index].w.data[i] = 0;
            _eulerian_fluid.fluid_bulk[index].w_valid.data[i] = 0;
          }
        }
      });

  delete[] resu;
  delete[] resv;
  delete[] resw;
  delete[] reswu;
  delete[] reswv;
  delete[] resww;
}

void FluidSim::compute_phi() {

  // make use of vdb
  //    openvdb::FloatGrid::Ptr levelset =
  //    vdbToolsWapper::particleToLevelset(particles, particle_radius, 0.5*dx);
  //    openvdb::tools::GridSampler<openvdb::FloatGrid,
  //    openvdb::tools::BoxSampler> box_sampler(*levelset);
  //
  // try atomic
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].global_index.data[i] =
                            index * 512 + i;
                      }
                    });

  int n = eulerian_fluids.n_bulks * eulerian_fluids.n_perbulk;
  float *res = new float[n];
  float *resw = new float[n];
  memset(res, 0, sizeof(float) * n);
  memset(resw, 0, sizeof(float) * n);
//    tbb::parallel_for((size_t)0,
//                      (size_t)eulerian_fluids.n_bulks,
//                      (size_t)1,
//                      [&](size_t index)
//                      {
//                          for (int i=0;i<eulerian_fluids.n_perbulk;i++)
//                          {
//                              int gidx =
//                              eulerian_fluids.fluid_bulk[index].global_index.data[i];
//                              res[gidx] = 3.0f*dx;
//                              eulerian_fluids.fluid_bulk[index].liquid_phi.data[i]
//                              = 3.0f*dx;
//
//                          }
//
//                      });
#pragma omp parallel for
  for (int i = 0; i < particles.size(); i++) {
    FLUID::Vec3f pos = particles[i].pos - (eulerian_fluids.bmin +
                                           dx * FLUID::Vec3f(0.5, 0.5, 0.5));

    FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos / (8 * eulerian_fluids.h));
    FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos / eulerian_fluids.h);
    FLUID::Vec3i particle_local_ijk = particle_ijk - 8 * bulk_ijk;
    for (int kk = particle_local_ijk[2] - 1; kk <= particle_local_ijk[2] + 1;
         kk++)
      for (int jj = particle_local_ijk[1] - 1; jj <= particle_local_ijk[1] + 1;
           jj++)
        for (int ii = particle_local_ijk[0] - 1;
             ii <= particle_local_ijk[0] + 1; ii++) {
          int bulkidx = eulerian_fluids.find_bulk(
              particle_ijk[0], particle_ijk[1], particle_ijk[2]);
          int mem_idx = eulerian_fluids.global_index(bulkidx, ii, jj, kk);

          FLUID::Vec3f sample_pos =
              dx * FLUID::Vec3f(8 * bulk_ijk + FLUID::Vec3i(ii, jj, kk));

          float weight0 = FluidSim::compute_weight(sample_pos, pos, dx);
#pragma omp atomic
          res[mem_idx] -= weight0 * particle_radius;
#pragma omp atomic
          resw[mem_idx] += (float)weight0;
        }
  }
#pragma omp parallel for
  for (int i = 0; i < n; i++) {

    res[i] = res[i] / (1e-6 + resw[i]);
    // std::cout<<res[i]<<std::endl;
  }
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          int gidx = eulerian_fluids.fluid_bulk[index].global_index.data[i];

          eulerian_fluids.fluid_bulk[index].liquid_phi.data[i] =
              0.5 * dx + 2.0 * res[gidx];
        }
      });
  //    tbb::parallel_for((size_t)0,
  //                      (size_t)eulerian_fluids.n_bulks,
  //                      (size_t)1,
  //                      [&](size_t index)
  //                      {
  //                          for (int i=0;i<eulerian_fluids.n_perbulk;i++)
  //                          {
  //                              int gidx =
  //                              eulerian_fluids.fluid_bulk[index].global_index.data[i];
  //                              if(res[gidx]==0)
  //                              {
  //                                  float value = dx;
  //                                  FLUID::Vec3i ijk =
  //                                  eulerian_fluids.loop_order[i]; for(int kk
  //                                  = ijk[2]-1;kk<=ijk[2]+1;kk++)for(int jj =
  //                                  ijk[1]-1;jj<=ijk[1]+1;jj++)for(int ii =
  //                                  ijk[0]-1; ii<=ijk[0]+1; ii++)
  //                                          {
  //                                              if(eulerian_fluids.liquid_phi(index,
  //                                              ii,jj,kk)<0)
  //                                              {
  //                                                  value = min(value,
  //                                                  dx+eulerian_fluids.liquid_phi(index,
  //                                                  ii,jj,kk));
  //                                              }
  //                                          }
  //
  //
  //                              }
  //
  //                          }
  //
  //                      });

  //    tbb::parallel_for((size_t)0,
  //                      (size_t)eulerian_fluids.fluid_bulk.size(),
  //                      (size_t)1,
  //                      [&](size_t index)
  //                      {
  //                          for (uint i=0;i<eulerian_fluids.n_perbulk;i++)
  //                          {
  //                              FLUID::Vec3i ijk =
  //                              eulerian_fluids.loop_order[i]; FLUID::Vec3i
  //                              bulk_ijk =
  //                              eulerian_fluids.fluid_bulk[index].tile_corner;
  //                              FLUID::Vec3f sample_pos = eulerian_fluids.bmin
  //                              +
  //                              dx*FLUID::Vec3f(bulk_ijk+ijk)+dx*FLUID::Vec3f(0.5,0.5,0.5);
  //                              float sdf_value =
  //                              box_sampler.wsSample(openvdb::Vec3R(sample_pos[0],
  //                              sample_pos[1], sample_pos[2]));
  //                              eulerian_fluids.liquid_phi(index,
  //                              ijk[0],ijk[1],ijk[2]) = 3.0*dx;
  //                              if(sdf_value<3.0*dx)
  //                                  eulerian_fluids.liquid_phi(index,
  //                                  ijk[0],ijk[1],ijk[2]) = sdf_value;
  //                              eulerian_fluids.liquid_phi(index,
  //                              ijk[0],ijk[1],ijk[2]) += particle_radius;
  //
  //
  //                          }
  //
  //                      });
  //
  //    for(int i=0;i<particles.size();i++)
  //    {
  //        FLUID::Vec3f pos = particles[i].pos
  //                    - eulerian_fluids.bmin;
  //        FLUID::Vec3i bulk_ijk = FLUID::Vec3i(pos/(8*eulerian_fluids.h));
  //        FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos/eulerian_fluids.h);
  //        FLUID::Vec3i particle_local_ijk = particle_ijk - 8*bulk_ijk;
  //        for (int
  //        kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
  //            for (int
  //            jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++)
  //                for (int
  //                ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
  //                {
  //                    int bulkidx =
  //                    eulerian_fluids.find_bulk(particle_ijk[0],particle_ijk[1],particle_ijk[2]);
  //                    int mem_idx =
  //                    eulerian_fluids.global_index(bulkidx,ii,jj,kk);
  //                    if(mem_idx>=0&&mem_idx<n)
  //                    {
  //                        FLUID::Vec3f sample_pos =
  //                        dx*FLUID::Vec3f(8*bulk_ijk+FLUID::Vec3i(ii,jj,kk))+dx*FLUID::Vec3f(0.5,0.5,0.5);
  //                        {
  //                            float liquid_phi = res[mem_idx];
  //                            float test = dist(sample_pos, pos) -
  //                            particle_radius; if(test<liquid_phi) {
  //                                res[mem_idx] = test;
  //                            }
  //                        }
  //
  //                    }
  //
  //                }
  //    }
  //    tbb::parallel_for((size_t)0,
  //                      (size_t)eulerian_fluids.n_bulks,
  //                      (size_t)1,
  //                      [&](size_t index)
  //                      {
  //                          for (int i=0;i<eulerian_fluids.n_perbulk;i++)
  //                          {
  //                              int gidx =
  //                              eulerian_fluids.fluid_bulk[index].global_index.data[i];
  //                              eulerian_fluids.fluid_bulk[index].liquid_phi.data[i]
  //                              = res[gidx];
  //
  //                          }
  //
  //                      });

  //	//provided all particles sorted in bulks
  //	//for all bulks in parallel, for all particles
  //	//in this bulk, spread their quantities to
  //	//the grid of this bulk.
  //	tbb::parallel_for((size_t)0,
  //		              (size_t)particle_bulks.size(),
  //					  (size_t)1,
  //					  [&](size_t index)
  //	{
  //
  //		for (uint i=0;i<particle_bulks[index].size();i++)
  //		{
  //			FLUID::Vec3f pos =
  // particles[particle_bulks[index][i]].pos
  //				- eulerian_fluids.bmin;
  //			FLUID::Vec3i bulk_ijk =
  // eulerian_fluids.fluid_bulk[index].tile_corner;
  // FLUID::Vec3i particle_ijk = FLUID::Vec3i(pos/eulerian_fluids.h);
  // FLUID::Vec3i particle_local_ijk =
  // particle_ijk - bulk_ijk; 			for (int
  // kk=particle_local_ijk[2]-1;kk<=particle_local_ijk[2]+1;kk++)
  // for (int jj=particle_local_ijk[1]-1;jj<=particle_local_ijk[1]+1;jj++) for
  // (int ii=particle_local_ijk[0]-1;ii<=particle_local_ijk[0]+1;ii++)
  //			{
  //				if (kk>=0&&kk<eulerian_fluids.tile_n
  //				  &&jj>=0&&jj<eulerian_fluids.tile_n
  //				  &&ii>=0&&ii<eulerian_fluids.tile_n)
  //				{
  //					float liquid_phi =
  // eulerian_fluids.liquid_phi(index,ii,jj,kk);
  // FLUID::Vec3f sample_pos =
  // dx*FLUID::Vec3f(bulk_ijk+FLUID::Vec3i(ii,jj,kk))+dx*FLUID::Vec3f(0.5,0.5,0.5);
  //					float test_val =
  // dist(sample_pos,pos)-particle_radius;
  // if(test_val<liquid_phi)
  //					{
  //						eulerian_fluids.liquid_phi(index,ii,jj,kk)
  //= test_val;
  //					}
  //				}
  //			}
  //		}
  //
  //	});

  // extend liquids slightly into solids
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.fluid_bulk.size(), (size_t)1,
      [&](size_t index) {
        for (uint i = 0; i < eulerian_fluids.n_perbulk; i++) {
          FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
          if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) <
              0.5 * dx) {
            float solid_phi =
                0.125 *
                (eulerian_fluids.solid_phi(index, ijk[0], ijk[1], ijk[2]) +
                 eulerian_fluids.solid_phi(index, ijk[0] + 1, ijk[1], ijk[2]) +
                 eulerian_fluids.solid_phi(index, ijk[0], ijk[1] + 1, ijk[2]) +
                 eulerian_fluids.solid_phi(index, ijk[0] + 1, ijk[1] + 1,
                                           ijk[2]) +
                 eulerian_fluids.solid_phi(index, ijk[0], ijk[1], ijk[2] + 1) +
                 eulerian_fluids.solid_phi(index, ijk[0] + 1, ijk[1],
                                           ijk[2] + 1) +
                 eulerian_fluids.solid_phi(index, ijk[0], ijk[1] + 1,
                                           ijk[2] + 1) +
                 eulerian_fluids.solid_phi(index, ijk[0] + 1, ijk[1] + 1,
                                           ijk[2] + 1));
            if (solid_phi < 0) {
              eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) =
                  -0.5 * dx;
            }
          }
        }
      });
  delete[] res;
  delete[] resw;
}
void FluidSim::set_boundary(float (*phi)(const FLUID::Vec3f &)) {
  tbb::parallel_for((uint)0, eulerian_fluids.n_bulks, (uint)1, [&](uint b) {
    for (int i = 0; i < eulerian_fluids.loop_order.size(); i++) {
      FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
      FLUID::Vec3f pos =
          eulerian_fluids.bmin +
          dx * FLUID::Vec3f(eulerian_fluids.fluid_bulk[b].tile_corner + ijk);
      eulerian_fluids.solid_phi(b, ijk[0], ijk[1], ijk[2]) = phi(pos);
    }
  });
}
void FluidSim::postAdvBoundary()
{
    ((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
            [&](size_t index) {
                for(int ii=0;ii<eulerian_fluids.n_perbulk;ii++)
                {
                    FLUID::Vec3i ijk = eulerian_fluids.loop_order[ii];
                    int i=ijk[0], j=ijk[1], k=ijk[2];
                    FLUID::Vec3f pos = eulerian_fluids.bmin + dx*FLUID::Vec3f(eulerian_fluids.fluid_bulk[index].tile_corner + FLUID::Vec3i(i,j,k));
                    if(eulerian_fluids.u_weight(index, i,j,k)>0)
                    {
                        eulerian_fluids.u_valid(index, i,j,k) = 1;
                    } else {
                        auto posu = pos + dx*FLUID::Vec3f(0,0.5,0.5);
                        if(eulerian_fluids.get_solid_phi(posu)<0)
                        {
                            eulerian_fluids.u(index, i,j,k) = eulerian_fluids.u_solid(index, i,j,k);
                            eulerian_fluids.u_valid(index, i,j,k) = 1;
                        } else {
                            eulerian_fluids.u_valid(index, i,j,k) = 0;
                        }
                    }

                    if(eulerian_fluids.v_weight(index, i,j,k)>0)
                    {
                        eulerian_fluids.v_valid(index, i,j,k) = 1;
                    } else {
                        auto posv = pos + dx*FLUID::Vec3f(0.5,0,0.5);
                        if(eulerian_fluids.get_solid_phi(posv)<0)
                        {
                            eulerian_fluids.v(index, i,j,k) = eulerian_fluids.v_solid(index, i,j,k);
                            eulerian_fluids.v_valid(index, i,j,k) = 1;
                        } else {
                            eulerian_fluids.v_valid(index, i,j,k) = 0;
                        }
                    }

                    if(eulerian_fluids.w_weight(index, i,j,k)>0)
                    {
                        eulerian_fluids.w_valid(index, i,j,k) = 1;
                    } else {
                        auto posw = pos + dx*FLUID::Vec3f(0.5,0.5,0);
                        if(eulerian_fluids.get_solid_phi(posw)<0)
                        {
                            eulerian_fluids.w(index, i,j,k) = eulerian_fluids.w_solid(index, i,j,k);
                            eulerian_fluids.w_valid(index, i,j,k) = 1;
                        } else {
                            eulerian_fluids.w_valid(index, i,j,k) = 0;
                        }
                    }
                }
            });
}
void FluidSim::advance(float dt, float (*phi)(const FLUID::Vec3f &)) {
  float t = 0;
  cfl_dt = 10.0 * cfl();
  while (t < dt) {
    float substep = cfl_dt;
    if (t + substep > dt)
      substep = dt - t;
    printf("Taking substep of size %f (to %0.3f%% of the frame)\n", substep,
           100 * (t + substep) / dt);

    printf("FLIP advection\n");
    emit(phi);
    emitRegion(phi,substep);
    FLIP_advection(substep);

    //		printf("Emission\n");
    //		emitFluids(substep, phi);

    printf("reinitialize bulks\n");
    eulerian_fluids.initialize_bulks(particles, dx);

    printf("setting boundary\n");
    set_boundary(phi);

    printf("particle to grid\n");
    // particle_to_grid(eulerian_fluids, particles, dx);
      tbb::parallel_for((size_t)0, (size_t)particles.size(), (size_t)1,
                        [&](size_t index) {
          particles[index].volm = 1.0;
                        });
    fusion_p2g_liquid_phi(eulerian_fluids, particles, dx, particle_radius);
    compute_weights();
    //postAdvBoundary();

    extrapolate(eulerian_fluids, 2);

    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                          eulerian_fluids.fluid_bulk[index].u_save.data[i] =
                              eulerian_fluids.fluid_bulk[index].u.data[i];
                          eulerian_fluids.fluid_bulk[index].v_save.data[i] =
                              eulerian_fluids.fluid_bulk[index].v.data[i];
                          eulerian_fluids.fluid_bulk[index].w_save.data[i] =
                              eulerian_fluids.fluid_bulk[index].w.data[i];
                        }
                      });
    // printf("computing phi\n");
    // compute_phi();

    printf("add gravity\n");
    add_force(substep);

    printf(" Pressure projection\n");
    project(substep);
    // Pressure projection only produces valid velocities in faces with non-zero
    // associated face area. Because the advection step may interpolate from
    // these invalid faces, we must extrapolate velocities from the fluid domain
    // into these invalid faces.
    printf(" Extrapolation\n");
    extrapolate(20);
    remeshing();
    // For extrapolated velocities, replace the normal component with
    // that of the object.
    printf(" Constrain boundary velocities\n");
    constrain_velocity();
    t += substep;
  }
  total_frame++;
}
void FluidSim::add_force(float dt) {
  tbb::parallel_for((uint)0, (uint)eulerian_fluids.n_bulks, (uint)1,
                    [&](uint b) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        // eulerian_fluids.fluid_bulk[b].v.data[i] -= dt*9.81f;
                        eulerian_fluids.fluid_bulk[b].v.data[i] += dt * gravity;
                      }
                    });
}
void FluidSim::compute_weights() {
  tbb::parallel_for((uint)0, eulerian_fluids.n_bulks, (uint)1, [&](uint b) {
    for (int i = 0; i < eulerian_fluids.loop_order.size(); i++) {
      FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
      FLUID::Vec3i bulk_corner = eulerian_fluids.fluid_bulk[b].tile_corner;

      eulerian_fluids.u_weight(b, ijk[0], ijk[1], ijk[2]) =
          1 - fraction_inside(
                  eulerian_fluids.solid_phi(b, ijk[0], ijk[1], ijk[2]),
                  eulerian_fluids.solid_phi(b, ijk[0], ijk[1] + 1, ijk[2]),
                  eulerian_fluids.solid_phi(b, ijk[0], ijk[1], ijk[2] + 1),
                  eulerian_fluids.solid_phi(b, ijk[0], ijk[1] + 1, ijk[2] + 1));
      eulerian_fluids.u_weight(b, ijk[0], ijk[1], ijk[2]) = clamp(
          eulerian_fluids.u_weight(b, ijk[0], ijk[1], ijk[2]), 0.0f, 1.0f);

      eulerian_fluids.v_weight(b, ijk[0], ijk[1], ijk[2]) =
          1 - fraction_inside(
                  eulerian_fluids.solid_phi(b, ijk[0], ijk[1], ijk[2]),
                  eulerian_fluids.solid_phi(b, ijk[0], ijk[1], ijk[2] + 1),
                  eulerian_fluids.solid_phi(b, ijk[0] + 1, ijk[1], ijk[2]),
                  eulerian_fluids.solid_phi(b, ijk[0] + 1, ijk[1], ijk[2] + 1));
      eulerian_fluids.v_weight(b, ijk[0], ijk[1], ijk[2]) = clamp(
          eulerian_fluids.v_weight(b, ijk[0], ijk[1], ijk[2]), 0.0f, 1.0f);

      eulerian_fluids.w_weight(b, ijk[0], ijk[1], ijk[2]) =
          1 - fraction_inside(
                  eulerian_fluids.solid_phi(b, ijk[0], ijk[1], ijk[2]),
                  eulerian_fluids.solid_phi(b, ijk[0], ijk[1] + 1, ijk[2]),
                  eulerian_fluids.solid_phi(b, ijk[0] + 1, ijk[1], ijk[2]),
                  eulerian_fluids.solid_phi(b, ijk[0] + 1, ijk[1] + 1, ijk[2]));
      eulerian_fluids.w_weight(b, ijk[0], ijk[1], ijk[2]) = clamp(
          eulerian_fluids.w_weight(b, ijk[0], ijk[1], ijk[2]), 0.0f, 1.0f);
    }
  });
}

FLUID::Vec3f FluidSim::trace_rk3(const FLUID::Vec3f &position, float dt) {
  	float c1 = 0.22222222222*dt, c2 = 0.33333333333 * dt, c3 = 0.44444444444*dt;
  	FLUID::Vec3f input = position;
  	FLUID::Vec3f velocity1 = getVelocity(input);
  	FLUID::Vec3f midp1 = input + ((float)(0.5*dt))*velocity1;
  	FLUID::Vec3f velocity2 = getVelocity(midp1);
  	FLUID::Vec3f midp2 = input + ((float)(0.75*dt))*velocity2;
  	FLUID::Vec3f velocity3 = getVelocity(midp2);
  	//velocity = get_velocity(input + 0.5f*dt*velocity);
  	//input += dt*velocity;
  	input = input + c1*velocity1 + c2*velocity2 + c3*velocity3;
//
//  FLUID::Vec3f input = position;
//  FLUID::Vec3f vel1 = getVelocity(input);
//  FLUID::Vec3f pos1 = input + 0.5f * dt * vel1;
//  FLUID::Vec3f vel2 = getVelocity(pos1);
//  FLUID::Vec3f pos2 = input + 0.5f * dt * vel2;
//  FLUID::Vec3f vel3 = getVelocity(pos2);
//  FLUID::Vec3f pos3 = input + dt * vel3;
//  FLUID::Vec3f vel4 = getVelocity(pos3);
//
//  input = input + 1.0f / 6.0f * dt * (vel1 + 2.0f * vel2 + 2.0f * vel3 + vel4);

  return input;
}
void FluidSim::particle_interpolate(float alpha) {
  // p.v = alpha * Interp(v) + (1-alpha)*(p.v + Interp(dv));
  tbb::parallel_for(
      (size_t)0, (size_t)particles.size(), (size_t)1, [&](size_t index) {
        FLUID::Vec3f pos = particles[index].pos;
        FLUID::Vec3f pv = particles[index].vel;
        FLUID::Vec3f v = eulerian_fluids.get_velocity(pos);
        FLUID::Vec3f dv = eulerian_fluids.get_delta_vel(pos);
        particles[index].vel = alpha * v + (1.0f - alpha) * (pv + dv);
      });
}

void FluidSim::constrain_velocity() {

  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].u_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].u.data[i];
                        eulerian_fluids.fluid_bulk[index].v_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].v.data[i];
                        eulerian_fluids.fluid_bulk[index].w_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].w.data[i];
                      }
                    });

  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
          FLUID::Vec3f pos(eulerian_fluids.fluid_bulk[index].tile_corner + ijk);
          if (eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2]) == 0) {
            FLUID::Vec3f posu =
                (pos + FLUID::Vec3f(0, 0.5, 0.5)) * dx + eulerian_fluids.bmin;
            FLUID::Vec3f velu = eulerian_fluids.get_velocity(posu);
            FLUID::Vec3f normalu(0, 0, 0);
            normalu = eulerian_fluids.get_grad_solid(posu);
            normalize(normalu);
            float perp_componentu = dot(velu, normalu);
            velu -= perp_componentu * normalu;
            eulerian_fluids.u_delta(index, ijk[0], ijk[1], ijk[2]) = velu[0];
          }
          if (eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2]) == 0) {
            FLUID::Vec3f posv =
                (pos + FLUID::Vec3f(0.5, 0, 0.5)) * dx + eulerian_fluids.bmin;
            FLUID::Vec3f velv = eulerian_fluids.get_velocity(posv);
            FLUID::Vec3f normalv(0, 0, 0);
            normalv = eulerian_fluids.get_grad_solid(posv);
            normalize(normalv);
            float perp_componentv = dot(velv, normalv);
            velv -= perp_componentv * normalv;
            eulerian_fluids.v_delta(index, ijk[0], ijk[1], ijk[2]) = velv[1];
          }
          if (eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2]) == 0) {
            FLUID::Vec3f posw =
                (pos + FLUID::Vec3f(0.5, 0.5, 0)) * dx + eulerian_fluids.bmin;
            FLUID::Vec3f velw = eulerian_fluids.get_velocity(posw);
            FLUID::Vec3f normalw(0, 0, 0);
            normalw = eulerian_fluids.get_grad_solid(posw);
            normalize(normalw);
            float perp_componentw = dot(velw, normalw);
            velw -= perp_componentw * normalw;
            eulerian_fluids.w_delta(index, ijk[0], ijk[1], ijk[2]) = velw[2];
          }
        }
      });

  // update
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].u.data[i] =
                            eulerian_fluids.fluid_bulk[index].u_delta.data[i];
                        eulerian_fluids.fluid_bulk[index].v.data[i] =
                            eulerian_fluids.fluid_bulk[index].v_delta.data[i];
                        eulerian_fluids.fluid_bulk[index].w.data[i] =
                            eulerian_fluids.fluid_bulk[index].w_delta.data[i];
                      }
                    });
}
void FluidSim::solve_pressure_parallel_build(float dt) {
  vector<FLUID::Vec3i> Bulk_ijk;
  Bulk_ijk.resize(eulerian_fluids.n_bulks);
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        Bulk_ijk[index] = eulerian_fluids.fluid_bulk[index].tile_corner / 8;
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          eulerian_fluids.fluid_bulk[index].global_index.data[i] =
              index * 512 + i;
          eulerian_fluids.fluid_bulk[index].pressure.data[i] = 0.0;
        }
      });
  Dofs.resize(512 * eulerian_fluids.n_bulks);

  std::cout << "PPE unkowns:" << Dofs.size() << std::endl;
  matrix.resize(Dofs.size());
  rhs.resize(Dofs.size());
  matrix.zero();
  rhs.assign(rhs.size(), 0);
  Dofs.assign(Dofs.size(), 0);
  std::cout << "begin assemble" << std::endl;
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
          float centre_phi =
              eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]);
          if (centre_phi < 0) {
            uint Dof_idx =
                eulerian_fluids.global_index(index, ijk[0], ijk[1], ijk[2]);
            // right neighbour
            float term =
                eulerian_fluids.u_weight(index, ijk[0] + 1, ijk[1], ijk[2]) *
                dt / sqr(dx);

            float right_phi =
                eulerian_fluids.liquid_phi(index, ijk[0] + 1, ijk[1], ijk[2]);
            if (right_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0] + 1, ijk[1], ijk[2]),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, right_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
            }
            rhs[Dof_idx] -=
                eulerian_fluids.u_weight(index, ijk[0] + 1, ijk[1], ijk[2]) *
                eulerian_fluids.u(index, ijk[0] + 1, ijk[1], ijk[2]) / dx;

            // left neighbour
            term = eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2]) *
                   dt / sqr(dx);

            float left_phi =
                eulerian_fluids.liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]);
            if (left_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0] - 1, ijk[1], ijk[2]),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, left_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
            }
            rhs[Dof_idx] +=
                eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2]) *
                eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2]) / dx;

            // top neighbour
            term = eulerian_fluids.v_weight(index, ijk[0], ijk[1] + 1, ijk[2]) *
                   dt / sqr(dx);
            float top_phi =
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] + 1, ijk[2]);
            if (top_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0], ijk[1] + 1, ijk[2]),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, top_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
            }
            rhs[Dof_idx] -=
                eulerian_fluids.v_weight(index, ijk[0], ijk[1] + 1, ijk[2]) *
                eulerian_fluids.v(index, ijk[0], ijk[1] + 1, ijk[2]) / dx;

            // bottom neighbour
            term = eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2]) *
                   dt / sqr(dx);
            float bot_phi =
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]);
            if (bot_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0], ijk[1] - 1, ijk[2]),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, bot_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
            }
            rhs[Dof_idx] +=
                eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2]) *
                eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2]) / dx;

            // far neighbour
            term = eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2] + 1) *
                   dt / sqr(dx);
            float far_phi =
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] + 1);
            if (far_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0], ijk[1], ijk[2] + 1),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, far_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
            }
            rhs[Dof_idx] -=
                eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2] + 1) *
                eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2] + 1) / dx;

            // near neighbour
            term = eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2]) *
                   dt / sqr(dx);
            float near_phi =
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1);
            if (near_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0], ijk[1], ijk[2] - 1),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, near_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
            }
            rhs[Dof_idx] +=
                eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2]) *
                eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2]) / dx;

            if(eulerian_fluids.u_weight(index,ijk[0],  ijk[1],  ijk[2])==0 &&
               eulerian_fluids.u_weight(index,ijk[0]+1,ijk[1],  ijk[2])==0 &&
               eulerian_fluids.v_weight(index,ijk[0],  ijk[1],  ijk[2])==0 &&
               eulerian_fluids.v_weight(index,ijk[0],  ijk[1]+1,ijk[2])==0 &&
               eulerian_fluids.w_weight(index,ijk[0],  ijk[1],  ijk[2])==0 &&
               eulerian_fluids.w_weight(index,ijk[0],  ijk[1],  ijk[2]+1)==0)
            {
                rhs[Dof_idx] = 0;
            }
          }
        }
      });

  cout << "assign matrix done" << endl;
  FLUID::Vec3i nijk =
      FLUID::Vec3i((eulerian_fluids.bmax - eulerian_fluids.bmin) / dx);
  double tolerance;
  int iterations;
  bool success = AMGPCGSolveSparseParallelBuild(
      matrix, rhs, Dofs, 1e-12, 100, tolerance, iterations, Bulk_ijk);
  printf("Solver took %d iterations and had residual %e\n", iterations,
         tolerance);
  if (!success) {
    printf("WARNING: Pressure solve "
           "failed!************************************************\n");
  }

  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
          if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0) {
            eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) =
                Dofs[eulerian_fluids.global_index(index, ijk[0], ijk[1],
                                                  ijk[2])];
          } else {
            eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) = 0;
          }
        }
      });

  // u = u- grad p;
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].u_valid.data[i] = 0;
                        eulerian_fluids.fluid_bulk[index].v_valid.data[i] = 0;
                        eulerian_fluids.fluid_bulk[index].w_valid.data[i] = 0;
                      }
                    });

  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
          if (eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2]) > 0 &&
              (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0 ||
               eulerian_fluids.liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]) <
                   0)) {
            float theta = 1;
            if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) >=
                    0 ||
                eulerian_fluids.liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]) >=
                    0) {
              theta = fraction_inside(
                  eulerian_fluids.liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]),
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]));
            }
            if (theta < 0.01)
              theta = 0.01;
            eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2]) -=
                dt *
                ((eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) -
                  eulerian_fluids.pressure(index, ijk[0] - 1, ijk[1], ijk[2])) /
                 dx / theta);
            eulerian_fluids.u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
            FLUID::Vec3f sample_pos_u =
                eulerian_fluids.bmin +
                dx * FLUID::Vec3f(
                         eulerian_fluids.fluid_bulk[index].tile_corner + ijk) +
                dx * FLUID::Vec3f(0.0, 0.5, 0.5);
            if (eulerian_fluids.get_liquid_phi(sample_pos_u) >
                0) // particularly important for numerical stability
            {
              eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2]) = 0;
              eulerian_fluids.u_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
            }

          } else {
            eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2]) = 0;
            eulerian_fluids.u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
          }

          if (eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2]) > 0 &&
              (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0 ||
               eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]) <
                   0)) {
            float theta = 1;
            if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) >=
                    0 ||
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]) >=
                    0) {
              theta = fraction_inside(
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]),
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]));
            }
            if (theta < 0.01)
              theta = 0.01;
            eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2]) -=
                dt *
                ((eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) -
                  eulerian_fluids.pressure(index, ijk[0], ijk[1] - 1, ijk[2])) /
                 dx / theta);
            eulerian_fluids.v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
            FLUID::Vec3f sample_pos_v =
                eulerian_fluids.bmin +
                dx * FLUID::Vec3f(
                         eulerian_fluids.fluid_bulk[index].tile_corner + ijk) +
                dx * FLUID::Vec3f(0.5, 0.0, 0.5);
            if (eulerian_fluids.get_liquid_phi(sample_pos_v) >
                0) // particularly important for numerical stability
            {
              eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2]) = 0;
              eulerian_fluids.v_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
            }
          } else {
            eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2]) = 0;
            eulerian_fluids.v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
          }

          if (eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2]) > 0 &&
              (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0 ||
               eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1) <
                   0)) {
            float theta = 1;
            if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) >=
                    0 ||
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1) >=
                    0) {
              theta = fraction_inside(
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1),
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]));
            }
            if (theta < 0.01)
              theta = 0.01;
            eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2]) -=
                dt *
                ((eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) -
                  eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2] - 1)) /
                 dx / theta);
            eulerian_fluids.w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
            FLUID::Vec3f sample_pos_w =
                eulerian_fluids.bmin +
                dx * FLUID::Vec3f(
                         eulerian_fluids.fluid_bulk[index].tile_corner + ijk) +
                dx * FLUID::Vec3f(0.5, 0.5, 0.0);
            if (eulerian_fluids.get_liquid_phi(sample_pos_w) >
                0) // particularly important for numerical stability
            {
              eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2]) = 0;
              eulerian_fluids.w_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
            }
          } else {
            eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2]) = 0;
            eulerian_fluids.w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
          }
        }
      });

  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          if (eulerian_fluids.fluid_bulk[index].u_valid.data[i] == 0) {
            eulerian_fluids.fluid_bulk[index].u.data[i] = 0;
          }
          if (eulerian_fluids.fluid_bulk[index].v_valid.data[i] == 0) {
            eulerian_fluids.fluid_bulk[index].v.data[i] = 0;
          }
          if (eulerian_fluids.fluid_bulk[index].v_valid.data[i] == 0) {
            eulerian_fluids.fluid_bulk[index].v.data[i] = 0;
          }
        }
      });
}
void FluidSim::boundaryModel(float dt, float nu, std::vector<FLIP_particle> &_p, float (*phi)(const FLUID::Vec3f &)) {

    tbb::parallel_for((size_t)0, (size_t)_p.size(), (size_t)1, [&](size_t index){
        float dist = phi(_p[index].pos);
        if(dist>=0&&dist<2.0*dx)
        {

            FLUID::Vec3i ijk = FLUID::Vec3i((_p[index].pos - eulerian_fluids.bmin)/eulerian_fluids.h);
            float weight = 0.000001;
            FLUID::Vec3f sum(0);
            for(int kk = ijk[2]-2;kk<=ijk[2]+2;kk++)
            {
                for(int jj=ijk[1]-2;jj<=ijk[1]+2;jj++)
                {
                    for(int ii=ijk[0]-2;ii<=ijk[0]+2;ii++)
                    {
                        FLUID::Vec3f pos = eulerian_fluids.bmin + dx*FLUID::Vec3f(ii,jj,kk) + dx*FLUID::Vec3f(0.5);
                        FLUID::Vec3f dvel = eulerian_fluids.get_velocity(pos)-_p[index].vel;
                        FLUID::Vec3f dir = _p[index].pos - pos;
                        float w = 1.0f/pow(4.0f*M_PI*nu*dt, 1.5)*exp(-FLUID::mag2(dir)/(4.0f*nu*dt));
                        weight += w;
                        sum += w * dvel;
                    }
                }
            }
            _p[index].vel += sum/weight;

        }
    });
}
void FluidSim::solve_pressure(float dt) {
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          eulerian_fluids.fluid_bulk[index].global_index.data[i] = 0;
          eulerian_fluids.fluid_bulk[index].pressure.data[i] = 0.0;
        }
      });
  Dofs.resize(0);
  vector<FLUID::Vec3i> Dof_ijk;
  Dof_ijk.resize(0);
  for (size_t index = 0; index < eulerian_fluids.n_bulks; index++) {
    for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
      FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
      if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0) {
        eulerian_fluids.global_index(index, ijk[0], ijk[1], ijk[2]) =
            Dofs.size();
        Dofs.push_back(0);
        Dof_ijk.push_back(eulerian_fluids.fluid_bulk[index].tile_corner + ijk);
      }
    }
  }
  std::cout << "PPE unkowns:" << Dofs.size() << std::endl;
  matrix.resize(Dofs.size());
  rhs.resize(Dofs.size());
  matrix.zero();
  rhs.assign(rhs.size(), 0);
  Dofs.assign(Dofs.size(), 0);

    bool pure_neumann = true;
  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
          float centre_phi =
              eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]);
          if (centre_phi < 0) {
            uint Dof_idx =
                eulerian_fluids.global_index(index, ijk[0], ijk[1], ijk[2]);
            // right neighbour
            float term =
                eulerian_fluids.u_weight(index, ijk[0] + 1, ijk[1], ijk[2]) *
                dt / sqr(dx);

            float right_phi =
                eulerian_fluids.liquid_phi(index, ijk[0] + 1, ijk[1], ijk[2]);
            if (right_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0] + 1, ijk[1], ijk[2]),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, right_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
              pure_neumann = false;
            }
            rhs[Dof_idx] -=
                (eulerian_fluids.u_weight(index, ijk[0] + 1, ijk[1], ijk[2]) *
                     eulerian_fluids.u(index, ijk[0] + 1, ijk[1], ijk[2]) +
                 (1.0f -
                  eulerian_fluids.u_weight(index, ijk[0] + 1, ijk[1], ijk[2])) *
                     eulerian_fluids.u_solid(index, ijk[0] + 1, ijk[1],
                                             ijk[2])) /
                dx;

            // left neighbour
            term = eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2]) *
                   dt / sqr(dx);

            float left_phi =
                eulerian_fluids.liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]);
            if (left_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0] - 1, ijk[1], ijk[2]),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, left_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
                pure_neumann = false;
            }

            rhs[Dof_idx] +=
                (eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2]) *
                     eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2]) +
                 (1.0 -
                  eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2])) *
                     eulerian_fluids.u_solid(index, ijk[0], ijk[1], ijk[2])) /
                dx;

            // top neighbour
            term = eulerian_fluids.v_weight(index, ijk[0], ijk[1] + 1, ijk[2]) *
                   dt / sqr(dx);
            float top_phi =
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] + 1, ijk[2]);
            if (top_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0], ijk[1] + 1, ijk[2]),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, top_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
                pure_neumann = false;
            }
            rhs[Dof_idx] -=
                (eulerian_fluids.v_weight(index, ijk[0], ijk[1] + 1, ijk[2]) *
                     eulerian_fluids.v(index, ijk[0], ijk[1] + 1, ijk[2]) +
                 (1.0 -
                  eulerian_fluids.v_weight(index, ijk[0], ijk[1] + 1, ijk[2])) *
                     eulerian_fluids.v_solid(index, ijk[0], ijk[1] + 1,
                                             ijk[2])) /
                dx;

            // bottom neighbour
            term = eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2]) *
                   dt / sqr(dx);
            float bot_phi =
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]);
            if (bot_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0], ijk[1] - 1, ijk[2]),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, bot_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
                pure_neumann = false;
            }
            rhs[Dof_idx] +=
                (eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2]) *
                     eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2]) +
                 (1.0 -
                  eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2])) *
                     eulerian_fluids.v_solid(index, ijk[0], ijk[1], ijk[2])) /
                dx;

            // far neighbour
            term = eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2] + 1) *
                   dt / sqr(dx);
            float far_phi =
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] + 1);
            if (far_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0], ijk[1], ijk[2] + 1),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, far_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
                pure_neumann = false;
            }
            rhs[Dof_idx] -=
                (eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2] + 1) *
                     eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2] + 1) +
                 (1.0 -
                  eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2] + 1)) *
                     eulerian_fluids.w_solid(index, ijk[0], ijk[1],
                                             ijk[2] + 1)) /
                dx;

            // near neighbour
            term = eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2]) *
                   dt / sqr(dx);
            float near_phi =
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1);
            if (near_phi < 0) {
              matrix.add_to_element(Dof_idx, Dof_idx, term);
              matrix.add_to_element(Dof_idx,
                                    eulerian_fluids.global_index(
                                        index, ijk[0], ijk[1], ijk[2] - 1),
                                    -term);
            } else {
              float theta = fraction_inside(centre_phi, near_phi);
              if (theta < 0.01f)
                theta = 0.01f;
              matrix.add_to_element(Dof_idx, Dof_idx, term / theta);
                pure_neumann = false;
            }
            rhs[Dof_idx] +=
                (eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2]) *
                     eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2]) +
                 (1.0 -
                  eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2])) *
                     eulerian_fluids.w_solid(index, ijk[0], ijk[1], ijk[2])) /
                dx;

          if(eulerian_fluids.u_weight(index,ijk[0],  ijk[1],  ijk[2])==0 &&
             eulerian_fluids.u_weight(index,ijk[0]+1,ijk[1],  ijk[2])==0 &&
             eulerian_fluids.v_weight(index,ijk[0],  ijk[1],  ijk[2])==0 &&
             eulerian_fluids.v_weight(index,ijk[0],  ijk[1]+1,ijk[2])==0 &&
             eulerian_fluids.w_weight(index,ijk[0],  ijk[1],  ijk[2])==0 &&
             eulerian_fluids.w_weight(index,ijk[0],  ijk[1],  ijk[2]+1)==0)
          {
              rhs[Dof_idx] = 0;
          }
          }
        }
      });






  cout << "assign matrix done" << endl;
  FLUID::Vec3i nijk =
      FLUID::Vec3i((eulerian_fluids.bmax - eulerian_fluids.bmin) / dx);
  double tolerance;
  int iterations;
  bool success =
      AMGPCGSolveSparse(matrix, rhs, Dofs, Dof_ijk, 1e-6, 100, tolerance,
                        iterations, nijk[0], nijk[1], nijk[2], pure_neumann);
  printf("Solver took %d iterations and had residual %e\n", iterations,
         tolerance);
  if (!success) {
    printf("WARNING: Pressure solve "
           "failed!************************************************\n");
  }

  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
          if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0) {
            eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) =
                Dofs[eulerian_fluids.global_index(index, ijk[0], ijk[1],
                                                  ijk[2])];
          } else {
            eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) = 0;
          }
        }
      });

  // u = u- grad p;
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].u_valid.data[i] = 0;
                        eulerian_fluids.fluid_bulk[index].v_valid.data[i] = 0;
                        eulerian_fluids.fluid_bulk[index].w_valid.data[i] = 0;
                      }
                    });

  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];
          if (eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2]) > 0 &&
              (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0 ||
               eulerian_fluids.liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]) <
                   0))
          {
            float theta = 1;
            if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) >=
                    0 ||
                eulerian_fluids.liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]) >=
                    0) {
              theta = fraction_inside(
                  eulerian_fluids.liquid_phi(index, ijk[0] - 1, ijk[1], ijk[2]),
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]));
            }
            if (theta < 0.01)
              theta = 0.01;
            eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2]) -=
                dt *
                ((eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) -
                  eulerian_fluids.pressure(index, ijk[0] - 1, ijk[1], ijk[2])) /
                 dx / theta);
            eulerian_fluids.u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
            FLUID::Vec3f sample_pos_u =
                eulerian_fluids.bmin +
                dx * FLUID::Vec3f(
                         eulerian_fluids.fluid_bulk[index].tile_corner + ijk) +
                dx * FLUID::Vec3f(0.0, 0.5, 0.5);
            if (eulerian_fluids.get_liquid_phi(sample_pos_u) >
                0) // particularly important for numerical stability
            {
              // eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2]) = 0;
              eulerian_fluids.u_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
            }

          } else  {
            eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2]) =
                eulerian_fluids.u_solid(index, ijk[0], ijk[1], ijk[2]);
            if(eulerian_fluids.u_weight(index, ijk[0], ijk[1], ijk[2])<1){
                eulerian_fluids.u_valid(index, ijk[0], ijk[1], ijk[2])=1;
            }
            else{
                eulerian_fluids.u_valid(index, ijk[0], ijk[1], ijk[2])=0;
            }

          }

          if (eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2]) > 0 &&
              (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0 ||
               eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]) <
                   0)) {
            float theta = 1;
            if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) >=
                    0 ||
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]) >=
                    0) {
              theta = fraction_inside(
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1] - 1, ijk[2]),
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]));
            }
            if (theta < 0.01)
              theta = 0.01;
            eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2]) -=
                dt *
                ((eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) -
                  eulerian_fluids.pressure(index, ijk[0], ijk[1] - 1, ijk[2])) /
                 dx / theta);
            eulerian_fluids.v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
            FLUID::Vec3f sample_pos_v =
                eulerian_fluids.bmin +
                dx * FLUID::Vec3f(
                         eulerian_fluids.fluid_bulk[index].tile_corner + ijk) +
                dx * FLUID::Vec3f(0.5, 0.0, 0.5);
            if (eulerian_fluids.get_liquid_phi(sample_pos_v) >
                0) // particularly important for numerical stability
            {
              // eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2]) = 0;
              eulerian_fluids.v_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
            }
          } else {
            eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2]) =
                eulerian_fluids.v_solid(index, ijk[0], ijk[1], ijk[2]);

            if(eulerian_fluids.v_weight(index, ijk[0], ijk[1], ijk[2])<1){
                eulerian_fluids.v_valid(index, ijk[0], ijk[1], ijk[2])=1;
            }
            else{
                eulerian_fluids.v_valid(index, ijk[0], ijk[1], ijk[2])=0;
            }
          }

          if (eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2]) > 0 &&
              (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) < 0 ||
               eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1) <
                   0)) {
            float theta = 1;
            if (eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]) >=
                    0 ||
                eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1) >=
                    0) {
              theta = fraction_inside(
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2] - 1),
                  eulerian_fluids.liquid_phi(index, ijk[0], ijk[1], ijk[2]));
            }
            if (theta < 0.01)
              theta = 0.01;
            eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2]) -=
                dt *
                ((eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2]) -
                  eulerian_fluids.pressure(index, ijk[0], ijk[1], ijk[2] - 1)) /
                 dx / theta);
            eulerian_fluids.w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
            FLUID::Vec3f sample_pos_w =
                eulerian_fluids.bmin +
                dx * FLUID::Vec3f(
                         eulerian_fluids.fluid_bulk[index].tile_corner + ijk) +
                dx * FLUID::Vec3f(0.5, 0.5, 0.0);
            if (eulerian_fluids.get_liquid_phi(sample_pos_w) >
                0) // particularly important for numerical stability
            {
              // eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2]) = 0;
              eulerian_fluids.w_valid(index, ijk[0], ijk[1], ijk[2]) = 0;
            }
          } else {
            eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2]) =
                eulerian_fluids.w_solid(index, ijk[0], ijk[1], ijk[2]);

              if(eulerian_fluids.w_weight(index, ijk[0], ijk[1], ijk[2])<1){
                  eulerian_fluids.w_valid(index, ijk[0], ijk[1], ijk[2])=1;
              }
              else{
                  eulerian_fluids.w_valid(index, ijk[0], ijk[1], ijk[2])=0;
              }
          }
        }
      });

  tbb::parallel_for(
      (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1, [&](size_t index) {
        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
          if (eulerian_fluids.fluid_bulk[index].u_valid.data[i] == 0) {
            eulerian_fluids.fluid_bulk[index].u.data[i] = 0;
          }
          if (eulerian_fluids.fluid_bulk[index].v_valid.data[i] == 0) {
            eulerian_fluids.fluid_bulk[index].v.data[i] = 0;
          }
          if (eulerian_fluids.fluid_bulk[index].w_valid.data[i] == 0) {
            eulerian_fluids.fluid_bulk[index].w.data[i] = 0;
          }
        }
      });
}
void FluidSim::project(float dt) {
  // Compute finite-volume type face area weight for each velocity sample.
  //compute_weights();

  // Set up and solve the variational pressure solve.
  solve_pressure(dt);
}
void FluidSim::project(float substep, std::vector<FLIP_particle> &p, float order_coef, float (*phi)(const FLUID::Vec3f &))
{

    //particles.swap(p);
    printf("reinitialize bulks\n");
    eulerian_fluids.initialize_bulks(p, dx);

    printf("setting boundary\n");
    set_boundary(phi);

    printf("particle to grid\n");
    // particle_to_grid(eulerian_fluids, particles, dx);
    fusion_p2g_liquid_phi(eulerian_fluids, p, dx, particle_radius);
    compute_weights();
    postAdvBoundary();
//
    if(EmitterSampleField!= nullptr)
    {
        tbb::parallel_for((size_t) 0, (size_t) eulerian_fluids.n_bulks, (size_t) 1,
                          [&](size_t index) {
                              for (int o = 0; o < eulerian_fluids.n_perbulk; o++) {
                                  FLUID::Vec3i ijk = eulerian_fluids.loop_order[o];
                                  int i=ijk[0],j=ijk[1],k=ijk[2];
                                  FLUID::Vec3f pos = eulerian_fluids.bmin + dx*FLUID::Vec3f(ijk) + dx*FLUID::Vec3f(eulerian_fluids.fluid_bulk[index].tile_corner) + dx*FLUID::Vec3f(0.5);
                                  FLUID::Vec3f posu = pos-dx*FLUID::Vec3f(0.5,0,0);
                                  FLUID::Vec3f posv = pos-dx*FLUID::Vec3f(0,0.5,0);
                                  FLUID::Vec3f posw = pos-dx*FLUID::Vec3f(0,0,0.5);
                                  if(eulerian_fluids.u_valid(index, i,j,k)==0)
                                  {
                                      eulerian_fluids.u(index,i,j,k) = EmitterSampleField->getVelocity(posu)[0];
                                  }
                                  if(eulerian_fluids.v_valid(index, i,j,k)==0)
                                  {
                                      eulerian_fluids.v(index,i,j,k) = EmitterSampleField->getVelocity(posv)[1];
                                  }
                                  if(eulerian_fluids.w_valid(index, i,j,k)==0)
                                  {
                                      eulerian_fluids.w(index,i,j,k) = EmitterSampleField->getVelocity(posw)[2];
                                  }
                              }
                          });
    } else{
        extrapolate(eulerian_fluids, 2);
    }

//     tbb::parallel_for((size_t) 0, (size_t) eulerian_fluids.n_bulks, (size_t) 1,
//                       [&](size_t index) {
//                           for (int o = 0; o < eulerian_fluids.n_perbulk; o++) {
//                               FLUID::Vec3i ijk = eulerian_fluids.loop_order[o];
//                               int i=ijk[0],j=ijk[1],k=ijk[2];
//                               FLUID::Vec3f pos = eulerian_fluids.bmin + dx*FLUID::Vec3f(ijk) + dx*FLUID::Vec3f(eulerian_fluids.fluid_bulk[index].tile_corner) + dx*FLUID::Vec3f(0.5);
//                               if(eulerian_fluids.get_solid_phi(pos)>0)
//                               {
//                                   for(auto e:Emitters)
//                                   {
//                                       if(sampleEmitter(pos, e)<0)
//                                       {
//                                           eulerian_fluids.u(index, i-1,j,k)= e.vel[0];
//                                           eulerian_fluids.u(index, i,  j,k)= e.vel[0];
//                                           eulerian_fluids.v(index, i,j-1,k)= e.vel[1];
//                                           eulerian_fluids.v(index, i,j,  k)= e.vel[1];
//                                           eulerian_fluids.w(index, i,j,k-1)= e.vel[2];
//                                           eulerian_fluids.w(index, i,j,k  )= e.vel[2];
//                                           eulerian_fluids.u_weight(index, i-1,j,k) = 0;
//                                           eulerian_fluids.u_weight(index, i,  j,k) = 0;
//                                           eulerian_fluids.v_weight(index, i,j-1,k) = 0;
//                                           eulerian_fluids.v_weight(index, i,j,  k) = 0;
//                                           eulerian_fluids.w_weight(index, i,j,k-1) = 0;
//                                           eulerian_fluids.w_weight(index, i,j,k  ) = 0;
//                                       }
//                                   }
//                               }
//                           }
//                       });
    tbb::parallel_for((size_t) 0, (size_t) eulerian_fluids.n_bulks, (size_t) 1,
                      [&](size_t index) {
                          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                              eulerian_fluids.fluid_bulk[index].u_save.data[i] =
                                      eulerian_fluids.fluid_bulk[index].u.data[i];
                              eulerian_fluids.fluid_bulk[index].v_save.data[i] =
                                      eulerian_fluids.fluid_bulk[index].v.data[i];
                              eulerian_fluids.fluid_bulk[index].w_save.data[i] =
                                      eulerian_fluids.fluid_bulk[index].w.data[i];
                          }
                      });

     // printf("computing phi\n");
     // compute_phi();

     printf("add gravity\n");
     add_force(substep);

     printf(" Pressure projection\n");
     project(substep);

    // Pressure projection only produces valid velocities in faces with non-zero
    // associated face area. Because the advection step may interpolate from
    // these invalid faces, we must extrapolate velocities from the fluid domain
    // into these invalid faces.
    printf(" Extrapolation\n");
    extrapolate(10);
    // For extrapolated velocities, replace the normal component with
    // that of the object.
    printf(" Constrain boundary velocities\n");
    constrain_velocity();
    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                              eulerian_fluids.fluid_bulk[index].u_delta.data[i] = 0;
                          }
                          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                              eulerian_fluids.fluid_bulk[index].v_delta.data[i] = 0;
                          }
                          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                              eulerian_fluids.fluid_bulk[index].w_delta.data[i] = 0;
                          }
                      });

    // compute delta,
    // for all eulerian bulks, compute u_delta
    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                              eulerian_fluids.fluid_bulk[index].u_delta.data[i] =
                                      eulerian_fluids.fluid_bulk[index].u.data[i] -
                                      eulerian_fluids.fluid_bulk[index].u_save.data[i];

                          }
                          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                              eulerian_fluids.fluid_bulk[index].v_delta.data[i] =
                                      eulerian_fluids.fluid_bulk[index].v.data[i] -
                                      eulerian_fluids.fluid_bulk[index].v_save.data[i];

                          }
                          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                              eulerian_fluids.fluid_bulk[index].w_delta.data[i] =
                                      eulerian_fluids.fluid_bulk[index].w.data[i] -
                                      eulerian_fluids.fluid_bulk[index].w_save.data[i];

                          }
                      });

    // for each particle, p.vel = alpha*Interp(U) + (1-alpha)*(U_p + Interp(dU))
    //particle_interpolate(0);
    //particles.swap(p);

}
void FluidSim::extrapolate(int times) {
  tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                        eulerian_fluids.fluid_bulk[index].u_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].u.data[i];
                        eulerian_fluids.fluid_bulk[index].v_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].v.data[i];
                        eulerian_fluids.fluid_bulk[index].w_delta.data[i] =
                            eulerian_fluids.fluid_bulk[index].w.data[i];
                      }
                    });
  // extrapolate u
  for (int layer = 0; layer < times; layer++) {
    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                          eulerian_fluids.fluid_bulk[index].old_valid.data[i] =
                              eulerian_fluids.fluid_bulk[index].u_valid.data[i];
                        }
                      });

    tbb::parallel_for(
        (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
        [&](size_t index) {
          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
            FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];

            if (eulerian_fluids.old_valid(index, ijk[0], ijk[1], ijk[2]) != 1) {
              int count = 0;
              float sum = 0;

              if (eulerian_fluids.old_valid(index, ijk[0] + 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.u(index, ijk[0] + 1, ijk[1], ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0] - 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.u(index, ijk[0] - 1, ijk[1], ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1] + 1,
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.u(index, ijk[0], ijk[1] + 1, ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1] - 1,
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.u(index, ijk[0], ijk[1] - 1, ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] + 1) == 1) {
                count++;
                sum += eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2] + 1);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] - 1) == 1) {
                count++;
                sum += eulerian_fluids.u(index, ijk[0], ijk[1], ijk[2] - 1);
              }

              if (count > 0) {
                eulerian_fluids.u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                eulerian_fluids.u_delta(index, ijk[0], ijk[1], ijk[2]) =
                    sum / (float)count;
              }
            }
          }
        });
    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                          eulerian_fluids.fluid_bulk[index].u.data[i] =
                              eulerian_fluids.fluid_bulk[index].u_delta.data[i];
                        }
                      });
  }

  // extrapolate v
  for (int layer = 0; layer < times; layer++) {
    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                          eulerian_fluids.fluid_bulk[index].old_valid.data[i] =
                              eulerian_fluids.fluid_bulk[index].v_valid.data[i];
                        }
                      });

    tbb::parallel_for(
        (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
        [&](size_t index) {
          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
            FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];

            if (eulerian_fluids.old_valid(index, ijk[0], ijk[1], ijk[2]) != 1) {
              int count = 0;
              float sum = 0;

              if (eulerian_fluids.old_valid(index, ijk[0] + 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.v(index, ijk[0] + 1, ijk[1], ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0] - 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.v(index, ijk[0] - 1, ijk[1], ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1] + 1,
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.v(index, ijk[0], ijk[1] + 1, ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1] - 1,
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.v(index, ijk[0], ijk[1] - 1, ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] + 1) == 1) {
                count++;
                sum += eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2] + 1);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] - 1) == 1) {
                count++;
                sum += eulerian_fluids.v(index, ijk[0], ijk[1], ijk[2] - 1);
              }

              if (count > 0) {
                eulerian_fluids.v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                eulerian_fluids.v_delta(index, ijk[0], ijk[1], ijk[2]) =
                    sum / (float)count;
              }
            }
          }
        });
    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                          eulerian_fluids.fluid_bulk[index].v.data[i] =
                              eulerian_fluids.fluid_bulk[index].v_delta.data[i];
                        }
                      });
  }

  // extrapolate w
  for (int layer = 0; layer < times; layer++) {
    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                          eulerian_fluids.fluid_bulk[index].old_valid.data[i] =
                              eulerian_fluids.fluid_bulk[index].w_valid.data[i];
                        }
                      });

    tbb::parallel_for(
        (size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
        [&](size_t index) {
          for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
            FLUID::Vec3i ijk = eulerian_fluids.loop_order[i];

            if (eulerian_fluids.old_valid(index, ijk[0], ijk[1], ijk[2]) != 1) {
              int count = 0;
              float sum = 0;

              if (eulerian_fluids.old_valid(index, ijk[0] + 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.w(index, ijk[0] + 1, ijk[1], ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0] - 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.w(index, ijk[0] - 1, ijk[1], ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1] + 1,
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.w(index, ijk[0], ijk[1] + 1, ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1] - 1,
                                            ijk[2]) == 1) {
                count++;
                sum += eulerian_fluids.w(index, ijk[0], ijk[1] - 1, ijk[2]);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] + 1) == 1) {
                count++;
                sum += eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2] + 1);
              }
              if (eulerian_fluids.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] - 1) == 1) {
                count++;
                sum += eulerian_fluids.w(index, ijk[0], ijk[1], ijk[2] - 1);
              }

              if (count > 0) {
                eulerian_fluids.w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                eulerian_fluids.w_delta(index, ijk[0], ijk[1], ijk[2]) =
                    sum / (float)count;
              }
            }
          }
        });
    tbb::parallel_for((size_t)0, (size_t)eulerian_fluids.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < eulerian_fluids.n_perbulk; i++) {
                          eulerian_fluids.fluid_bulk[index].w.data[i] =
                              eulerian_fluids.fluid_bulk[index].w_delta.data[i];
                        }
                      });
  }
}
void FluidSim::extrapolate(sparse_fluid8x8x8 &_eulerian_fluid, int times) {
  tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                    [&](size_t index) {
                      for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                        _eulerian_fluid.fluid_bulk[index].u_delta.data[i] =
                            _eulerian_fluid.fluid_bulk[index].u.data[i];
                        _eulerian_fluid.fluid_bulk[index].v_delta.data[i] =
                            _eulerian_fluid.fluid_bulk[index].v.data[i];
                        _eulerian_fluid.fluid_bulk[index].w_delta.data[i] =
                            _eulerian_fluid.fluid_bulk[index].w.data[i];
                      }
                    });
  // extrapolate u
  for (int layer = 0; layer < times; layer++) {
    tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                          _eulerian_fluid.fluid_bulk[index].old_valid.data[i] =
                              _eulerian_fluid.fluid_bulk[index].u_valid.data[i];
                        }
                      });

    tbb::parallel_for(
        (size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
        [&](size_t index) {
          for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
            FLUID::Vec3i ijk = _eulerian_fluid.loop_order[i];

            if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1], ijk[2]) != 1) {
              int count = 0;
              float sum = 0;

              if (_eulerian_fluid.old_valid(index, ijk[0] + 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.u(index, ijk[0] + 1, ijk[1], ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0] - 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.u(index, ijk[0] - 1, ijk[1], ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1] + 1,
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.u(index, ijk[0], ijk[1] + 1, ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1] - 1,
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.u(index, ijk[0], ijk[1] - 1, ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] + 1) == 1) {
                count++;
                sum += _eulerian_fluid.u(index, ijk[0], ijk[1], ijk[2] + 1);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] - 1) == 1) {
                count++;
                sum += _eulerian_fluid.u(index, ijk[0], ijk[1], ijk[2] - 1);
              }

              if (count > 0) {
                _eulerian_fluid.u_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                _eulerian_fluid.u_delta(index, ijk[0], ijk[1], ijk[2]) =
                    sum / (float)count;
              }
            }
          }
        });
    tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                          _eulerian_fluid.fluid_bulk[index].u.data[i] =
                              _eulerian_fluid.fluid_bulk[index].u_delta.data[i];
                        }
                      });
  }

  // extrapolate v
  for (int layer = 0; layer < times; layer++) {
    tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                          _eulerian_fluid.fluid_bulk[index].old_valid.data[i] =
                              _eulerian_fluid.fluid_bulk[index].v_valid.data[i];
                        }
                      });

    tbb::parallel_for(
        (size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
        [&](size_t index) {
          for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
            FLUID::Vec3i ijk = _eulerian_fluid.loop_order[i];

            if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1], ijk[2]) != 1) {
              int count = 0;
              float sum = 0;

              if (_eulerian_fluid.old_valid(index, ijk[0] + 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.v(index, ijk[0] + 1, ijk[1], ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0] - 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.v(index, ijk[0] - 1, ijk[1], ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1] + 1,
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.v(index, ijk[0], ijk[1] + 1, ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1] - 1,
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.v(index, ijk[0], ijk[1] - 1, ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] + 1) == 1) {
                count++;
                sum += _eulerian_fluid.v(index, ijk[0], ijk[1], ijk[2] + 1);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] - 1) == 1) {
                count++;
                sum += _eulerian_fluid.v(index, ijk[0], ijk[1], ijk[2] - 1);
              }

              if (count > 0) {
                _eulerian_fluid.v_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                _eulerian_fluid.v_delta(index, ijk[0], ijk[1], ijk[2]) =
                    sum / (float)count;
              }
            }
          }
        });
    tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                          _eulerian_fluid.fluid_bulk[index].v.data[i] =
                              _eulerian_fluid.fluid_bulk[index].v_delta.data[i];
                        }
                      });
  }

  // extrapolate w
  for (int layer = 0; layer < times; layer++) {
    tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                          _eulerian_fluid.fluid_bulk[index].old_valid.data[i] =
                              _eulerian_fluid.fluid_bulk[index].w_valid.data[i];
                        }
                      });

    tbb::parallel_for(
        (size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
        [&](size_t index) {
          for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
            FLUID::Vec3i ijk = _eulerian_fluid.loop_order[i];

            if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1], ijk[2]) != 1) {
              int count = 0;
              float sum = 0;

              if (_eulerian_fluid.old_valid(index, ijk[0] + 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.w(index, ijk[0] + 1, ijk[1], ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0] - 1, ijk[1],
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.w(index, ijk[0] - 1, ijk[1], ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1] + 1,
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.w(index, ijk[0], ijk[1] + 1, ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1] - 1,
                                            ijk[2]) == 1) {
                count++;
                sum += _eulerian_fluid.w(index, ijk[0], ijk[1] - 1, ijk[2]);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] + 1) == 1) {
                count++;
                sum += _eulerian_fluid.w(index, ijk[0], ijk[1], ijk[2] + 1);
              }
              if (_eulerian_fluid.old_valid(index, ijk[0], ijk[1],
                                            ijk[2] - 1) == 1) {
                count++;
                sum += _eulerian_fluid.w(index, ijk[0], ijk[1], ijk[2] - 1);
              }

              if (count > 0) {
                _eulerian_fluid.w_valid(index, ijk[0], ijk[1], ijk[2]) = 1;
                _eulerian_fluid.w_delta(index, ijk[0], ijk[1], ijk[2]) =
                    sum / (float)count;
              }
            }
          }
        });
    tbb::parallel_for((size_t)0, (size_t)_eulerian_fluid.n_bulks, (size_t)1,
                      [&](size_t index) {
                        for (int i = 0; i < _eulerian_fluid.n_perbulk; i++) {
                          _eulerian_fluid.fluid_bulk[index].w.data[i] =
                              _eulerian_fluid.fluid_bulk[index].w_delta.data[i];
                        }
                      });
  }
}
