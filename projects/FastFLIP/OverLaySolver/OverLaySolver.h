//
// Created by zhxx on 2021/1/4.
//

#ifndef LARGEFLIP_OVERLAYSOLVER_H
#define LARGEFLIP_OVERLAYSOLVER_H
#include "../FLIP/fluidsim.h"


class Aero {
    using GridT =
    typename openvdb::Grid<typename openvdb::tree::Tree4<float, 5, 4, 3>::Type>;
    using TreeT = typename GridT::TreeType;
public:
    bool initialized = false;
    sparse_fluid8x8x8 eulerian_fluids;
    std::vector<FluidSim*> domains;
    tbb::concurrent_vector<FLIP_particle> tracers;
    std::vector<FLIP_particle> otracers;
    std::vector<typename GridT::Ptr> tracerEmitters;
    float dx;
    float gravity;
    void setGravity(float g=-9.81) {gravity = g;}
    FluidSim* getLatestDomain(){ return domains[domains.size()-1]; }
    float sampleEmitter(FLUID::Vec3f &pos, typename GridT::Ptr grid) {
        float phi = 1.f;
        openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
                grid->constTree(), grid->transform());
        openvdb::math::Vec3<float> P(pos[0], pos[1], pos[2]);
        return (float)(interpolator.wsSample(P)); // ws denotes world space

    }
    void emitTracers(float (*phi)(const FLUID::Vec3f &))
    {
        std::cout<<"tracers num:"<<otracers.size()<<std::endl;
        if (otracers.size()>0)
        {
            eulerian_fluids.initialize_bulks(otracers, dx);
            FluidSim::fusion_p2g_liquid_phi(eulerian_fluids,otracers, dx, dx*1.01*sqrt(3.0)/2.0);
        }
	

	
        for (auto &grid : tracerEmitters) {
              FluidSim::levelsetEmitter lse;
              lse.ls = grid;
              lse.vel = FLUID::Vec3f(0);
              FluidSim::emit(otracers, lse, eulerian_fluids, nullptr, 0.5*dx, phi);


        }
	if(!initialized)
	{
		initialized = true;
	}
	
    }
    void advance(float dt, float (*phi)(const FLUID::Vec3f &))
    {
        advance(dt, 1.0, phi);
        //advance(0.5*dt, 1.0, phi);
        for(auto d:domains) {
            tbb::parallel_for((size_t) 0, (size_t) d->particles.size(), (size_t) 1, [&](size_t index) {

                for (auto dj:domains) {
                    for (auto e:dj->Emitters) {
                        if (FluidSim::sampleEmitter(d->particles[index].pos, e) < 0) {
                            d->particles[index].vel = e.vel;
                        }
                    }
                }
            });
        }

    }
    void advance(float dt, float order_coef, float (*phi)(const FLUID::Vec3f &))
    {

        if(domains.size()==1)
        {
            domains[0]->advance(dt,phi);
            //otracers = domains[0]->particles;

        }
        else {
            for (int i = 0; i < domains.size(); i++) {
                if (i == 0) {
                    domains[i]->setEmitterSampleField(nullptr);
                    domains[i]->emit(phi);
                } else {
                    domains[i]->setEmitterSampleField(domains[i - 1]);
                    domains[i]->emit(phi);
                }
                domains[i]->emitRegion(phi, dt);

            }
            for(auto d:domains) {
                domains[domains.size() - 1]->boundaryModel(dt, 0.0001, d->particles, phi);
            }
            for (auto d:domains) {
                tbb::parallel_for((size_t) 0, (size_t) d->particles.size(), (size_t) 1, [&](size_t index) {
                    d->particles[index].pos = getLatestDomain()->trace_rk3(d->particles[index].pos, dt);
                });
            }
            std::vector <FLIP_particle> particles(0);
            int cnt = 0;
            for (auto d:domains) {
                cnt += d->particles.size();
            }
            particles.resize(cnt);
            std::vector<float> pr(cnt);
            cnt = 0;
            if(domains.size()>=1) {
                domains[domains.size() - 1]->resolveParticleBoundaryCollision();
            }
            for (auto d:domains) {

                tbb::parallel_for((size_t) 0, (size_t) d->particles.size(), (size_t) 1, [&](size_t index) {
                    particles[cnt + index] = d->particles[index];
                    pr[cnt + index] = d->dx;
                });
                cnt += d->particles.size();
            }
            for (auto d:domains) {
                std::vector<FLIP_particle> vp;
                std::vector<char> mask(particles.size());
                tbb::parallel_for((size_t)0, (size_t)particles.size(), (size_t)1, [&](size_t index)
                {
                    particles[index].volm = d->dx*d->dx*d->dx;
                    if(pr[index]<=d->dx)
                    {
                        mask[index] = 1;
                    }else
                    {
                        mask[index] = 0;
                    }
                });
		        FluidSim::subset_particles(particles, vp, mask);

                d->project(dt, vp, 1.0, phi);
                std::cout<<"project with domain "<<d->dx<<" done\n";
                tbb::parallel_for((size_t) 0, (size_t) particles.size(), (size_t) 1, [&](size_t index) {
                    if (pr[index]<=d->dx)
                    {
                        particles[index].vel += d->getDelta(particles[index].pos);
                    }
                });
            }
            for(auto d:domains)
            {
                if((order_coef - 1.0f) > 0.000001)
                tbb::parallel_for((size_t) 0, (size_t) particles.size(), (size_t) 1, [&](size_t index) {
                    if (pr[index]<=d->dx)
                    {
                        particles[index].vel += (order_coef - 1.0f)*d->getDelta(particles[index].pos);
                    }
                });
            }
            cnt = 0;
            std::vector<char> mask(particles.size());
            for (auto d:domains) {
                mask.assign(particles.size(),0);
                tbb::parallel_for((size_t) 0, (size_t)particles.size(), (size_t) 1, [&](size_t index) {
                    if(domains[0]->inDomain(particles[index].pos) && pr[index]<=d->dx && phi(particles[index].pos)>=0)
                    {
                        mask[index] = 1;
                    }
                });
                FluidSim::subset_particles(particles, d->particles, mask);
            }

            //for auto d:domains d.remeshing();
            for (auto d:domains) {
                d->remeshing();
            }

        }
    }
};

#endif //LARGEFLIP_OVERLAYSOLVER_H
