#pragma once

#ifndef __ZENO_SOLVER_H__
#define __ZENO_SOLVER_H__

#include <string>
#include <windows.h>

namespace zensolver
{
    void flip_solve(
        HANDLE hPipeWrite,                //向主进程写入的管道句柄
        const std::string& init_fluid,    /*初始流体*/    //TODO: 后续会统一solver和zeno3的类型，届时可直接传递类型
        int size_fluid,
        const std::string& static_collider,   /*静态碰撞体*/
        int size_collider,
        const std::string& emission_source = "",   /*发射源*/
        int size_emission = 0,
        float accuracy = 0.08,     /*精度*/
        float timestep = 0.04,     /*时间步长*/
        float max_substep = 1,     /*最大子步数*/
        /*重力*/
        float gravity_x = 0.f,
        float gravity_y = -9.8f,
        float gravity_z = 0.f,
        /*发射源速度*/
        float emit_vx = 0.f,
        float emit_vy = 0.f,
        float emit_vz = 0.f,
        bool is_emission = true,                            /*是否发射*/
        float dynamic_collide_strength = 1.f,                 /*动态碰撞强度*/
        float density = 1000,                          /*密度*/
        float surface_tension = 0,       /*表面张力*/
        float viscosity = 0,            /*粘性*/
        float wall_viscosity = 0,        /*壁面粘性*/
        float wall_viscosityRange = 0,   /*壁面粘性作用范围*/
        char* curve_force = 0,           /*曲线力*/
        int n_curve_force = 0,
        int curve_endframe = 100,          /*曲线终止帧*/
        float curve_range = 1.1f,           /*曲线作用范围*/
        float preview_size = 0,          /*预览大小*/
        float preview_minVelocity = 0,   /*预览最小速度*/
        float preview_maxVelocity = 2.f,   /*预览最大速度*/
        char* FSD = 0,                  /*流固耦合*/
        int n_size_FSD = 0
        );
}

#endif