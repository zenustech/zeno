#pragma once

#include <zen/zen.h>
#include <vector>
#include "VDBGrid.h"

namespace zenbase{
float G_FRAME_TIME = 1.0/24.0f;
float G_FRAME_TIME_ELAPSED = 0.0;
bool TIME_SETP_INTEGRATED = false;
    struct SetFrameTime : zen::INode {
        virtual void apply() override {
            float num = std::get<float>(get_param("num"));
            G_FRAME_TIME = num;
        }
    };
    static int defSetFrameTime = zen::defNodeClass<SetFrameTime>("SetFrameTimeStep",
        { /* inputs: */ {
        }, /* outputs: */ {
        
        }, /* params: */ {
        {"float", "num", "0.0"},
        }, /* category: */ {
        "Keywords",
        }});
    struct GetFrameTime : zen::INode {
        virtual void apply() override {
            auto res = zen::IObject::make<tFloat>();
            res->num = G_FRAME_TIME;
            set_output("FrameTimeStep",res);      
        }
    };
    static int defGetFrameTime = zen::defNodeClass<GetFrameTime>("GetFrameTimeStep",
        { /* inputs: */ {
        }, 
        /* outputs: */ 
        {
            "FrameTimeStep",
        }, 
        /* params: */ 
        {
         
        }, /* category: */ {
            "Keywords",
        }});
    struct SetFrameTimeElapsed : zen::INode {
        virtual void apply() override {
            float num = std::get<float>(get_param("num"));
            G_FRAME_TIME_ELAPSED = num;
        }
    };
    static int defSetFrameTimeElapsed = zen::defNodeClass<SetFrameTimeElapsed>("SetFrameTimeElapsed",
        { /* inputs: */ {
        }, /* outputs: */ {
        
        }, /* params: */ {
        {"float", "num", "0.0"},
        }, /* category: */ {
        "Keywords",
        }});
    struct IntegrateFrameTime : zen::INode {
        virtual void apply() override {

            float desired_dt = 1.0/24.0f; 
            if(has_input("Desired_dt"))
            {
                desired_dt = get_input("Desired_dt")->as<tFloat>()->num;
            }
            
            float dt = desired_dt;
            if(G_FRAME_TIME_ELAPSED + dt > G_FRAME_TIME)
            {
                dt = G_FRAME_TIME - G_FRAME_TIME_ELAPSED;
            }
            auto out_dt = zen::IObject::make<tFloat>();
            out_dt->num = dt;
            if(!TIME_SETP_INTEGRATED)
            {
                G_FRAME_TIME_ELAPSED += dt;
                TIME_SETP_INTEGRATED = true;
            }
            
            set_output("dt",out_dt);
        }
    };
    static int defIntegrateFrameTime = zen::defNodeClass<IntegrateFrameTime>("IntegrateFrameTime",
        { /* inputs: */ {
            "Desired_dt", 
        }, /* outputs: */ {
            "dt",
        }, /* params: */ {
        {
        }, /* category: */ {
            "Keywords",
        }});

    struct GetFrameTimeElapsed : zen::INode {
        virtual void apply() override {
            auto res = zen::IObject::make<tFloat>();
            res->num = G_FRAME_TIME_ELAPSED;
            set_output("FrameTimeElapsed",res);      
        }
    };
    static int defGetFrameTimeElapsed = zen::defNodeClass<GetFrameTimeElapsed>("GetFrameTimeElapsed",
        { /* inputs: */ {
        }, 
        /* outputs: */ 
        {
            "FrameTimeElapsed",
        }, 
        /* params: */ 
        {
         
        }, /* category: */ {
            "Keywords",
        }});
}