/**
 * @file NonlinearSolverNodes.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief An example for nonlinear solver on zeno.
 * @version 0.1
 * @date 2022-02-04
 * 
 * @copyright Copyright (c) 2022  Ma Pengfei
 * 
 */

#include <AlgebraSolver/StdVector.h>
#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <AlgebraSolver/NonlinearSolver.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <loguru/loguru.hpp>


#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/ContextManaged.h>
#include <zeno/types/NumericObject.h>

using VectorType = StdVector<double,double>;



struct RawDoubleObject : zeno::IObject
{
    double * data = nullptr;
    int num = 0;
};


struct VectorTypeObject : zeno::IObject
{
    std::shared_ptr<VectorType> vector = nullptr;
};

struct ConvertRawdataToVecotor : zeno::INode {
    virtual void apply() override {

        auto x_raw = get_input<RawDoubleObject>("RawDoubleObject");

        auto vector_object = std::make_shared<VectorTypeObject>();
        vector_object->vector = std::make_shared<VectorType>();
        vector_object->vector->resize(x_raw->num);

        for (size_t i = 0; i < x_raw->num; i++)
        {
            vector_object->vector->data()[i] = x_raw->data[i];
        }
        
        set_output("VectorTypeObject", std::move(vector_object));
    }
};

ZENDEFNODE(ConvertRawdataToVecotor,
        { /* inputs: */ {
            "RawDoubleObject",
        }, /* outputs: */ {
            "VectorTypeObject",
        }, /* params: */ {
            // defl min max; defl min; defl
        }, /* category: */ {
            "Zentricle",
        }});

struct MakeRawDoubleObject : zeno::INode {
    virtual void apply() override {
        auto raw_double_array = std::make_shared<RawDoubleObject>();

        // auto num = get_param<int>("value");
        auto num = get_input<zeno::NumericObject>("num")->get<int>();

        auto data = (double*)malloc(num*sizeof(double));
        std::fill(data, data+num, 1.0);

        raw_double_array->num = num;
        raw_double_array->data = data;

        set_output("RawDoubleObject", std::move(raw_double_array));
    }
};

ZENDEFNODE(MakeRawDoubleObject,
        { /* inputs: */ {
            "num",
        }, /* outputs: */ {
        "RawDoubleObject",
        }, /* params: */ {
        // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});


struct NonlinearProblemObject : zeno::IObject, NonlinearProblem<VectorType>
{
    std::shared_ptr<zeno::FunctionObject> function = nullptr;

    virtual void Residual(const VectorType& x, VectorType& r){

        auto args = std::make_shared<zeno::DictObject>();
        auto rets = std::make_shared<zeno::DictObject>();

        CHECK_F(x.size() == r.size(), "Wrong size.");

        // TODO : x should be a pointer!!!
        // use std::shared_ptr<const VectorType> x; instead of const VectorType& x
        // to avoid variable constructions and destructions.
        args->lut["x"] = x;                                             // 封装
        rets->lut = function->call(args->lut);                          // 调用
        r = zeno::safe_any_cast<VectorType>(rets->lut["r"]);            // 解封

    }
};

// User defined 
struct CalculateResidual : zeno::INode {
    virtual void apply() override {
        
        auto args = get_input<zeno::DictObject>("args");
        auto rets = std::make_shared<zeno::DictObject>();

        auto x = zeno::safe_any_cast<VectorType>(args->lut.at("x"));
        // auto r = zeno::safe_any_cast<VectorType>(args->lut.at("x"));

        VectorType r;
        r.resize(x.size());
        CHECK_F(x.size() == r.size(), "Wrong size.");

        auto _x = flatten(x).data;
        auto _r = flatten(r).data;

        _r[0] = std::exp(2.0*_x[0])/2.0 - _x[1];
        _r[1] = _x[0]*_x[0] + _x[1]*_x[1]-1.0;

        rets->lut["r"] = r;
        set_output("rets", std::move(rets));
    }
};

ZENDEFNODE(CalculateResidual,
        { /* inputs: */ {
            "args",
        }, /* outputs: */ {
            "rets",
        }, /* params: */ {
        {},  // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});

struct MakeNonlinearProblemObject : zeno::INode {

    virtual void apply() override {
        auto nlp = std::make_shared<NonlinearProblemObject>();
        // TODO : input a function constructed with FuncBegin and FuncEnd
        if (has_input("function")) nlp->function = get_input<zeno::FunctionObject>("function");
        else LOG_F(WARNING, "no function input");
        set_output("NonlinearProblemObject", std::move(nlp));
    }
};

ZENDEFNODE(MakeNonlinearProblemObject,
        { /* inputs: */ {
            "function",
        }, /* outputs: */ {
            "NonlinearProblemObject",
        }, /* params: */ {
        {},  // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});


struct JFNKSolverObject : zeno::IObject
{
    std::shared_ptr<NewtonSolver<VectorType>> ns = nullptr;
};


struct MakeJFNKSolver : zeno::INode {

    virtual void apply() override {

        // TODO : Need parameters such as problem size, line search newton solver, bicgstab
        auto num = get_input<zeno::NumericObject>("num")->get<int>();
        auto bicgstab = std::make_shared<BiCGSTAB<VectorType>>(num);
        auto ns = std::make_shared<NewtonSolver<VectorType>>(bicgstab);
        auto method = ns->method();
        LOG_F(INFO, "using %s. ", method.c_str());

        auto jfnk_solver = std::make_shared<JFNKSolverObject>();
        jfnk_solver->ns = ns;

        set_output("JFNKSolverObject", std::move(jfnk_solver));
    }
};

ZENDEFNODE(MakeJFNKSolver,
        { /* inputs: */ {
            "num",
        }, /* outputs: */ {
            "JFNKSolverObject",
        }, /* params: */ {
        {},  // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});


struct PrintVectorTypeObject : zeno::INode {
    virtual void apply() override {
        auto x = get_input<VectorTypeObject>("VectorTypeObject")->vector;

        for (size_t i = 0; i < x->size(); i++)
        {
            printf("%lf\n", x->data()[i]);
        }
    }
};

ZENDEFNODE(PrintVectorTypeObject,
        { /* inputs: */ {
        "VectorTypeObject",
        }, /* outputs: */ {
        }, /* params: */ {
        // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});

struct SolveNonlinearProblem : zeno::INode {

    virtual void apply() override {

        auto jfnk_solver = get_input<JFNKSolverObject>("JFNKSolverObject");
        auto nlp = get_input<NonlinearProblemObject>("NonlinearProblemObject");

        auto x_vector_object = get_input<VectorTypeObject>("VectorTypeObject");
        auto x_vector_ptr = x_vector_object->vector;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // NOTE : we solve F(x) = b here. But b is not necessary. b is to be removed. 
        VectorType bb;
        bb.resize(x_vector_ptr->size());
        /////////////////////////////////////////////////////////////////////////////////////////////

        auto nonlinear_result = jfnk_solver->ns->Solve(nlp, *x_vector_ptr, bb);


        if ( nonlinear_result.first) {
            LOG_F(WARNING, "Noninear solver succeed");
            LOG_F(WARNING, "residual : %lf, iter : %d", nonlinear_result.second.first, nonlinear_result.second.second);
        }
        else LOG_F(WARNING, "Noninear solver failed");

        set_output("VectorTypeObject", std::move(x_vector_object));
    }
};


ZENDEFNODE(SolveNonlinearProblem,
        { /* inputs: */ {
            "NonlinearProblemObject", "JFNKSolverObject", "VectorTypeObject",
        }, /* outputs: */ {
            "VectorTypeObject",
        }, /* params: */ {
        {},  // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});
