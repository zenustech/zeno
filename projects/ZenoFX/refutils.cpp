#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include "zeno/utils/format.h"

namespace zeno {

    std::string preApplyRefs(const std::string& code, Graph* pGraph)
    {
        std::string parsed_code = code;
        std::map<std::string, std::string> refTovar, symbVals;
        int tmpid = 1;
        auto i = parsed_code.find("ref(");
        while (i != std::string::npos) {
            auto iend = parsed_code.find(')', i);
            if (iend == std::string::npos)
                return code;        //no match ')' still return orignal code.

            auto refstr = parsed_code.substr(i, iend - i + 1);
            if (refstr.empty())
                return code;        //ref is empty, return original code.

            std::string refvar, symval;
            if (refTovar.find(refstr) != refTovar.end())
            {
                refvar = refTovar[refstr];
                symval = symbVals[refvar];
            }
            else
            {
                //insert into the beginning of the code.
                do {
                    refvar = zeno::format("@ref{}", tmpid++);
                } while (parsed_code.find(refvar) != std::string::npos);

                refTovar[refstr] = refvar;

                auto ref = refstr.substr(4, iend - i - 4);
                //resolve reference to get const value.
                auto itParam = ref.find('/');
                if (itParam == std::string::npos) {
                    return code;
                }

                int dim = -1;
                std::string param;

                auto itDim = ref.find('/', itParam + 1);
                if (itDim != std::string::npos) {
                    param = ref.substr(itParam + 1, itDim - itParam - 1);
                    auto dimstr = ref.substr(itDim + 1);
                    if (dimstr == "x") dim = 0;
                    else if (dimstr == "y") dim = 1;
                    else if (dimstr == "z") dim = 2;
                    else if (dimstr == "w") dim = 3;
                    else return code;
                }
                else {
                    param = ref.substr(itParam + 1);
                }

                std::string ident = ref.substr(0, itParam);
                //ident可能已经被改编过的，类似于"6875f81e-aaa/334fbddf-CreateSphere"这种形式
                //然而editor的节点是在共享子图上的，其上的zfxcode引用并不包含上层子图节点的路径信息
                //ref只记录了最终端节点的id，比如上述例子的334fbddf-CreateSphere。
                //所以，唯有遍历当前Graph的节点，检查是否以ident结尾
                bool bFound = false;
                for (auto const& [ident_, inode] : pGraph->m_nodes) {
                    if (ident_.length() >= ident.length() &&
                        ident_.compare(ident_.length() - ident.length(), ident.length(), ident) == 0) {
                        ident = ident_;
                        bFound = true;
                        break;
                    }
                }

                if (!bFound)
                    return code;

                auto& pNode = pGraph->m_nodes[ident];
                if (pNode->requireInput(param)) {
                    bool bExist = false;
                    ParamPrimitive input = pNode->get_input_prim_param(param, &bExist);
                    
                    const zvariant& val = input.defl;
                    if (std::holds_alternative<float>(val)) {
                        auto v = std::get<float>(val);
                        symval = zeno::format("{}", v);
                    }
                    else if (std::holds_alternative<int>(val)) {
                        auto v = std::get<int>(val);
                        symval = zeno::format("{}", v);
                    }
                    else if (std::holds_alternative<std::string>(val)) {
                        auto v = std::get<std::string>(val);
                        symval = zeno::format("\"{}\"", v);
                    }
                    else if (std::holds_alternative<zeno::vec2f>(val)) {
                        auto v = std::get<zeno::vec2f>(val);
                        if (dim >= 0 && dim <= 1) {
                            symval = zeno::format("{}", v[dim]);
                        }
                        else {
                            symval = zeno::format("vec2({},{})", v[0], v[1]);
                        }
                    }
                    else if (std::holds_alternative<zeno::vec2i>(val)) {
                        auto v = std::get<zeno::vec2i>(val);
                        if (dim >= 0 && dim <= 1) {
                            symval = zeno::format("{}", v[dim]);
                        }
                        else {
                            symval = zeno::format("vec2({},{})", v[0], v[1]);
                        }
                    }
                    else if (std::holds_alternative<zeno::vec3f>(val)) {
                        auto v = std::get<zeno::vec3f>(val);
                        if (dim >= 0 && dim <= 2) {
                            symval = zeno::format("{}", v[dim]);
                        }
                        else {
                            symval = zeno::format("vec3({},{},{})", v[0], v[1], v[2]);
                        }
                    }
                    else if (std::holds_alternative<zeno::vec3i>(val)) {
                        auto v = std::get<zeno::vec3i>(val);
                        if (dim >= 0 && dim <= 2) {
                            symval = zeno::format("{}", v[dim]);
                        }
                        else {
                            symval = zeno::format("vec3({},{},{})", v[0], v[1], v[2]);
                        }
                    }
                    else if (std::holds_alternative<zeno::vec4f>(val)) {
                        auto v = std::get<zeno::vec4f>(val);
                        if (dim >= 0 && dim <= 3) {
                            symval = zeno::format("{}", v[dim]);
                        }
                        else {
                            symval = zeno::format("vec4({},{},{},{})", v[0], v[1], v[2], v[3]);
                        }
                    }
                    else if (std::holds_alternative<zeno::vec4i>(val)) {
                        auto v = std::get<zeno::vec4i>(val);
                        if (dim >= 0 && dim <= 3) {
                            symval = zeno::format("{}", v[dim]);
                        }
                        else {
                            symval = zeno::format("vec4({},{},{},{})", v[0], v[1], v[2], v[3]);
                        }
                    }
                    else {
                        //todo
                    }

                    symbVals[refvar] = symval;
                }
            }

            parsed_code.replace(i, iend - i + 1, symval);

            i = i + symval.length();
            i = parsed_code.find("ref(", i);
        }
        return parsed_code;
    }
}

