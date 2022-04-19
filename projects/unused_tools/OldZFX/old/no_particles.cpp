#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include "no_ParticlesObject.h"
#include "particles.h"

using namespace zeno;

struct MakeParticles : INode {
    virtual void apply() override {
        auto particles = std::make_shared<ParticlesObject>();
        auto nchannels = get_param<int>("nchannels");
        auto size = get_param<int>("size");
        particles->pars.set_nchannels(nchannels);
        particles->pars.resize(size);
        set_output("particles", std::move(particles));
    }
};

ZENDEFNODE(MakeParticles, {
    {},
    {"particles"},
    {{"int", "nchannels", "0"}, {"int", "size", "0"}},
    {"zenofx"},
});


struct ParticlesToPrimitive : INode {
    virtual void apply() override {
        auto prim = std::make_shared<PrimitiveObject>();
        auto pars = &get_input<ParticlesObject>("particles")->pars;
        prim->resize(pars->size());
        auto &prim_pos = prim->add_attr<vec3f>("pos");
        auto &prim_vel = prim->add_attr<vec3f>("vel");
        auto &prim_clr = prim->add_attr<vec3f>("clr");
        for (size_t i = 0; i < pars->size(); i++) {
            auto x = pars->channel(0)[i];
            auto y = pars->channel(1)[i];
            auto z = pars->channel(2)[i];
            auto u = pars->channel(3)[i];
            auto v = pars->channel(4)[i];
            auto w = pars->channel(5)[i];
            auto r = pars->channel(6)[i];
            auto g = pars->channel(7)[i];
            auto b = pars->channel(8)[i];
            prim_pos[i] = vec3f(x, y, z);
            prim_vel[i] = vec3f(u, v, w);
            prim_clr[i] = vec3f(r, g, b);
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesToPrimitive, {
    {"particles"},
    {"prim"},
    {},
    {"zenofx"},
});

struct PrimitiveToParticles : INode {
    virtual void apply() override {
        auto particles = std::make_shared<ParticlesObject>();
        auto pars = &particles->pars;
        auto prim = get_input<PrimitiveObject>("prim");
        pars->resize(prim->size());
        pars->set_nchannels(9);
        pars->set_channel_name(0, "pos.0");
        pars->set_channel_name(1, "pos.1");
        pars->set_channel_name(2, "pos.2");
        pars->set_channel_name(3, "vel.0");
        pars->set_channel_name(4, "vel.1");
        pars->set_channel_name(5, "vel.2");
        pars->set_channel_name(6, "clr.0");
        pars->set_channel_name(7, "clr.1");
        pars->set_channel_name(8, "clr.2");
        auto &prim_pos = prim->add_attr<vec3f>("pos");
        auto &prim_vel = prim->add_attr<vec3f>("vel");
        auto &prim_clr = prim->add_attr<vec3f>("clr");
        for (size_t i = 0; i < pars->size(); i++) {
            auto &x = pars->channel(0)[i];
            auto &y = pars->channel(1)[i];
            auto &z = pars->channel(2)[i];
            auto &u = pars->channel(3)[i];
            auto &v = pars->channel(4)[i];
            auto &w = pars->channel(5)[i];
            auto &r = pars->channel(6)[i];
            auto &g = pars->channel(7)[i];
            auto &b = pars->channel(8)[i];
            x = prim_pos[i][0];
            y = prim_pos[i][1];
            z = prim_pos[i][2];
            u = prim_vel[i][0];
            v = prim_vel[i][1];
            w = prim_vel[i][2];
            r = prim_clr[i][0];
            g = prim_clr[i][1];
            b = prim_clr[i][2];
        }
        set_output("particles", std::move(particles));
    }
};

ZENDEFNODE(PrimitiveToParticles, {
    {"prim"},
    {"particles"},
    {},
    {"zenofx"},
});
