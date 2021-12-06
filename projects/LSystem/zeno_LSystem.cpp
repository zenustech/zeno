#include "zeno/zeno.h"
#include "zeno/types/StringObject.h"
#include "zeno/types/PrimitiveObject.h"

#include "LSystem/R3Mesh.h"

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

namespace zeno
{
    struct LSysGenerator : zeno::IObject
    {
        const bool _isPlus;
        const int _iterations;
        const int _defaultCoefficient;
        const int _thickness;
        std::string _axiom;
        std::vector<std::pair<std::string, std::string>> _rules;

        LSysGenerator(
            const bool isPlus,
            const int iterations,
            const int defaultCoefficient,
            const int thickness,
            std::string axiom)
            : _isPlus(isPlus),
              _iterations{iterations},
              _defaultCoefficient{defaultCoefficient},
              _thickness{thickness},
              _axiom{axiom},
              _rules{} {}

        bool isPlus()
        {
            return _isPlus;
        }

        std::string getCode()
        {
            std::stringstream ss;
            ss << _iterations << '\n'
               << _defaultCoefficient << '\n'
               << _thickness << '\n'
               << _axiom << '\n';
            for (const auto &[ruleName, rule] : _rules)
            {
                ss << ruleName << '=' << rule << '\n';
            }
            ss << '@' << '\n';
            return ss.str();

        }

        void appendRule(const std::string &ruleName)
        {
            _rules.emplace_back(std::pair{ruleName, ""});
        }

        void appendOp(const std::string &op)
        {
            _rules[_rules.size() - 1].second.append(op);
        }
    };

    struct MakeLSysGenerator : zeno::INode
    {
        virtual void apply() override
        {
            auto isPlus = static_cast<bool>(get_param<int>("isPlus"));
            auto iterations = get_param<int>("iterations");
            auto defaultCoefficient = get_param<int>("defaultCoefficient");
            auto thickness = get_param<int>("thickness");
            auto axiom = get_param<std::string>("axiom");
            auto generator = std::make_shared<LSysGenerator>(
                isPlus, iterations, defaultCoefficient, thickness, axiom);
            std::cout << "code: " << generator->getCode() << '\n';
            set_output("generator", std::move(generator));
        }
    };

    ZENDEFNODE(
        MakeLSysGenerator,
        {
            {},
            {
                {"LSysGenerator", "generator"},
            },
            {
                {"int", "isPlus", "1"},
                {"int", "iterations", "0"},
                {"int", "defaultCoefficient", "0"},
                {"int", "thickness", "0"},
                {"string", "axiom", ""},
            },
            {
                "LSystem",
            },
        });

    struct AppendLSysRule : zeno::INode
    {
        virtual void apply() override
        {
            auto generator = get_input<zeno::LSysGenerator>("generator");
            auto ruleName = get_param<std::string>("ruleName");
            generator->appendRule(ruleName);
            std::cout << "code: " << generator->getCode() << '\n';
            set_output("generator", std::move(generator));
        }
    };

    ZENDEFNODE(
        AppendLSysRule,
        {
            {
                {"LSysGenerator", "generator"},
            },
            {
                {"LSysGenerator", "generator"},
            },
            {
                {"string", "ruleName", ""},
            },
            {
                "LSystem",
            },
        });

    
    std::string code =
        "5\n"
        "15\n"
        "20\n"
        "fA\n"
        "A=^f >(30) B\\\\B\\\\\\\\\\B\n"
        "B=[^^fL \\\\\\\\\\\\A L  ]\n"
        "L=[^(60) [*(.3)] +(50)*(.28)]\n"
        "L=[^(60)*(.3)]\n"
        "L=[&(70)*(.3)]\n"
        "@";
    





    static const std::unordered_map<std::string_view, const char> kCommandTable{
        {"turnLeft", '+'},
        {"turnRight", '-'},
        {"pitchDown", '&'},
        {"pitchUp", '^'},
        {"thicken", '<'},
        {"rollLeft", '\\'},
        {"rollRight", '/'},
        {"narrow", '>'},
        {"setReduction", '%'},
        {"setThickness", '='},
        {"turn180", '|'},
        {"drawLeaf", '*'},
        {"drawBranch", 'f'}, // {"drawBranch", 'F'}, also work
        {"goForward", 'g'},  // {"goForward", 'G'}, also work
        {"saveState", '['},
        {"restoreState", ']'},
    };

    struct AppendLSysOP : zeno::INode
    {
        virtual void apply() override
        {
            auto generator = get_input<zeno::LSysGenerator>("generator");
            auto command = get_param<std::string>("command");
            auto param = get_param<std::string>("param");
            if (command == "rule")
            {
                generator->appendOp(param);
            }
            else
            {
                auto iter = kCommandTable.find(command);
                if (iter != kCommandTable.end())
                {
                    if (param != "")
                    {
                        stringstream ss;
                        ss << iter->second << '(' << param << ')';
                        generator->appendOp(ss.str());
                    }
                    else
                    {
                        generator->appendOp(std::string{iter->second});
                    }
                }
            }
            std::cout << "code: " << generator->getCode() << '\n';
            set_output("generator", std::move(generator));
        }
    };

    ZENDEFNODE(
        AppendLSysOP,
        {
            {
                {"LSysGenerator", "generator"},
            },
            {
                {"LSysGenerator", "generator"},
            },
            {
                {"enum "
                 "rule "
                 "turnLeft "
                 "turnRight "
                 "pitchDown "
                 "pitchUp "
                 "thicken "
                 "rollLeft "
                 "rollRight "
                 "narrow "
                 "setReduction "
                 "setThickness "
                 "turn180 "
                 "drawLeaf "
                 "drawBranch "
                 "goForward "
                 "saveState "
                 "restoreState",
                 "command", "turnLeft"},
                {"string", "param", ""},
            },
            {
                "LSystem",
            },
        });

    struct ProceduralTree : zeno::INode
    {
        virtual void apply() override
        {
            auto generator = get_input<zeno::LSysGenerator>("generator");
            auto code = generator->getCode();
            auto isPlus = generator->isPlus();
            R3Mesh mesh;
            mesh.Tree(code, isPlus);

            auto prim = std::make_shared<zeno::PrimitiveObject>();
            auto &pos = prim->add_attr<zeno::vec3f>("pos");
            auto &uv = prim->add_attr<zeno::vec3f>("uv");
            auto &nrm = prim->add_attr<zeno::vec3f>("nrm");

            std::unordered_map<R3MeshVertex *, int> m;

            prim->resize(mesh.NVertices());
            //#pragma omp parallel for
            for (int i = 0; i < mesh.NVertices(); ++i)
            {
                const auto &v{mesh.Vertex(i)};

                const auto &p{v->position};
                pos[i] = zeno::vec3f(p.X(), p.Y(), p.Z());

                const auto &t{v->texcoords};
                uv[i] = zeno::vec3f(t.X(), t.Y(), 0.0);

                const auto &n{v->normal};
                nrm[i] = zeno::vec3f(n.X(), n.Y(), n.Z());

                m[mesh.Vertex(i)] = i;
            }
            prim->tris.resize(mesh.NFaces());
#pragma omp parallel for
            for (int i = 0; i < mesh.NFaces(); ++i)
            {
                const auto &f{mesh.Face(i)};
                auto t0 = m[f->vertices[0]];
                auto t1 = m[f->vertices[1]];
                auto t2 = m[f->vertices[2]];
                prim->tris[i] = zeno::vec3i(t0, t1, t2);
            }
            set_output("prim", std::move(prim));
        }
    };

    ZENDEFNODE(
        ProceduralTree,
        {
            {
                {"LSysGenerator", "generator"},
            },
            {
                {"primitive", "prim"},
            },
            {},
            {
                "LSystem",
            },
        });

    /*
    struct R3MeshToPrim : zeno::INode
    {
        virtual void apply() override
        {
            auto prim = std::make_shared<zeno::PrimitiveObject>();
            set_output("prim", prim);
        }
    };

    ZENDEFNODE(R3MeshToPrim,
                   {{
                        "R3Mesh",
                    },
                    {
                        "prim",
                    },
                    {},
                     {
                        "LSystem",
                    }});

    struct PrimToR3Mesh : zeno::INode
    {
        virtual void apply() override
        {

            auto prim = get_input<zeno::PrimitiveObject>("prim");


        }
    };
    ZENDEFNODE(PrimToR3Mesh,
               { {
                    "prim",
                },
                 {
                    "R3Mesh",
                },
                {},
                {
                    "LSystem",
                }});
    */
} // namespace zeno