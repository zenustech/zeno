#include <zeno/zeno.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/ContextManaged.h>
#include <iostream>

class NonlinearProblem {
    virtual void Residual(const double &x, double &r) = 0;
};

struct NonlinearProblemObject2 : zeno::IObject, NonlinearProblem
{
    std::shared_ptr<zeno::FunctionObject> function = nullptr;

    virtual void Residual(const double &x, double &r){

        auto args = std::make_shared<zeno::DictObject>();
        auto rets = std::make_shared<zeno::DictObject>();

        std::cout << " Before calling func. \n"; 
        rets->lut = function->call(args->lut);
        std::cout << " After calling func. \n"; 
    }
};

struct CalculateResidual2 : zeno::INode {
    
    virtual void apply() override {
        
        auto args = get_input<zeno::DictObject>("args");
        auto rets = std::make_shared<zeno::DictObject>();

        static int n = 0;

        n++;
        std::cout << " Calculate Residual for the \n" << n << "th time.\n"; 

        rets->lut["n"] = n;
        set_output("rets", std::move(rets));
    }
};

ZENDEFNODE(CalculateResidual2,
        { /* inputs: */ {
            "args",
        }, /* outputs: */ {
            "rets",
        }, /* params: */ {
        {},  // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});

struct MakeNonlinearProblemObject2 : zeno::INode {

    virtual void apply() override {
        auto nlp = std::make_shared<NonlinearProblemObject2>();
        // TODO : input a function constructed with FuncBegin and FuncEnd
        if (has_input("function")) nlp->function = get_input<zeno::FunctionObject>("function");
        // else LOG_F(WARNING, "no function input");
        set_output("NonlinearProblemObject2", std::move(nlp));
    }
};

ZENDEFNODE(MakeNonlinearProblemObject2,
        { /* inputs: */ {
            "function",
        }, /* outputs: */ {
            "NonlinearProblemObject2",
        }, /* params: */ {
        {},  // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});


struct JFNKSolverObject2 : zeno::IObject
{
    // std::shared_ptr<NewtonSolver<VectorType>> ns = nullptr;
};


struct MakeJFNKSolver2 : zeno::INode {

    virtual void apply() override {

        auto jfnk_solver = std::make_shared<JFNKSolverObject2>();
        // jfnk_solver->ns = ns;

        set_output("JFNKSolverObject2", std::move(jfnk_solver));
    }
};

ZENDEFNODE(MakeJFNKSolver2,
        { /* inputs: */ {
        }, /* outputs: */ {
            "JFNKSolverObject2",
        }, /* params: */ {
        {},  // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});


struct SolveNonlinearProblem2 : zeno::INode {

    virtual void apply() override {

        auto jfnk_solver = get_input<JFNKSolverObject2>("JFNKSolverObject2");
        auto nlp = get_input<NonlinearProblemObject2>("NonlinearProblemObject2");

        auto args = std::make_shared<zeno::DictObject>();
        auto rets = std::make_shared<zeno::DictObject>();
        

        auto function = get_input<zeno::FunctionObject>("function");
        for (size_t i = 0; i < 100; i++){
            std::cout << " Before calling func. \n"; 
            rets->lut = function->call(args->lut);
            std::cout << " After calling func. \n"; 
        }  
    }
};


ZENDEFNODE(SolveNonlinearProblem2,
        { /* inputs: */ {
            "JFNKSolverObject2", "NonlinearProblemObject2", "function",
        }, /* outputs: */ {
        }, /* params: */ {
        {},  // defl min max; defl min; defl
        }, /* category: */ {
        "Zentricle",
        }});
