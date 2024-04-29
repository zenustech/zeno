#include "Structures.hpp"
#include "zensim/io/MeshIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/variantswitch.h>

#include "zensim/io/Filesystem.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace zeno {

struct AssetConversion : INode {
    void apply() override {
        auto pathStr = get_input2<std::string>("path");
        char *p = nullptr;
        if (pathStr.empty())
            p = getenv("PATH");
        else
            p = pathStr.data();
        std::set<std::string> pathLocations;
        auto processTags = [](std::string tags, std::set<std::string> &res, std::string sep) {
            using Ti = RM_CVREF_T(std::string::npos);
            Ti st = tags.find_first_not_of(sep, 0);
            for (auto ed = tags.find_first_of(sep, st + 1); ed != std::string::npos;
                 ed = tags.find_first_of(sep, st + 1)) {
                res.insert(tags.substr(st, ed - st));
                st = tags.find_first_not_of(sep, ed);
                if (st == std::string::npos)
                    break;
            }
            if (st != std::string::npos && st < tags.size()) {
                res.insert(tags.substr(st));
            }
        };
#ifdef ZS_PLATFORM_WINDOWS
        processTags(p, pathLocations, ";");
        const std::string target = "blender.exe";
#else
        processTags(p, pathLocations, ":");
        const std::string target = "blender";
#endif

        auto inputFile = get_input2<std::string>("input_model_file");
        fs::path inputPath = inputFile;
        auto ext = inputPath.extension().string();
        auto outputPath = get_input2<std::string>("output_model_file");
        auto format = fs::path(outputPath).extension().string();
        if (format.empty())
            format = ".obj";
        else if (format != ".obj" && format != ".gltf" && format != ".fbx")
            throw std::runtime_error(fmt::format("unknown output file format [{}]\n", outputPath));
        if (outputPath.empty())
            outputPath = inputPath.parent_path().string() + "/" + inputPath.stem().string() + format;

        for (const auto &path : pathLocations) {
            fmt::print("iterate path: {}\n", path);
            fs::path loc = path + "/" + target;
            bool ifExist = false;
            try {
                ifExist = fs::exists(loc);
            } catch (const std::exception &e) {
                fmt::print("\tskipping path {} due to exception (e.g. inaccessibility).\n", path);
                continue;
            }
            if (ifExist) {
                /// write script to a file
                auto scriptPath = zs::abs_exe_directory() + "/" + "resource/GltfToObj.py";
                std::ofstream os;
                os.open(scriptPath);
                std::string script = "import bpy\n";
                script += fmt::format("bpy.ops.import_scene{}(filepath=\"{}\")\n", ext, inputFile);
                script +=
                    fmt::format("bpy.ops.export_scene{}(filepath=\"{}\", use_selection=True)\n", format, outputPath);
                os << script;
                os.close();

                /// execute this blender script
                auto cmdStr = fmt::format("{} -b -P {}", loc.string(), scriptPath);
                fmt::print("\tfound {} at \"{}\".\n\texecuting \"{}\"\n", target, path, cmdStr);
                fmt::print("\tscript:\n========\n{}\n========\n", script);

                std::system(cmdStr.c_str());
            }
        }
    }
};
ZENDEFNODE(AssetConversion,
           {/* inputs: */ {
                {"string", "path", ""}, {"string", "input_model_file", ""}, {"string", "output_model_file", ""}},
            /* outputs: */
            {},
            /* params: */
            {},
            /* category: */
            {
                "blender",
            }});

// treat poly as lines (strands)
void primLineify(PrimitiveObject *prim, bool with_uv) {

    {
        std::vector<int> scansum(prim->polys.size());
        auto redsum = parallel_exclusive_scan_sum(prim->polys.begin(), prim->polys.end(), scansum.begin(),
                                                  [&](auto &ind) { return ind[1] - 1; });
        int linebase = prim->lines.size();
        prim->lines.resize(linebase + redsum);

        if (!prim->loops.has_attr("uvs") || !with_uv) {
            parallel_for(prim->polys.size(), [&](size_t i) {
                auto [start, len] = prim->polys[i];
                int scanbase = linebase + scansum[i];
                for (int j = 0; j + 1 < len; ++j) {
                    prim->lines[scanbase++] = vec2i(prim->loops[start + j], prim->loops[start + j + 1]);
                }
            });

        } else {
            const auto &loop_uv = prim->loops.attr<int>("uvs");
            const auto &uvs = prim->uvs;
            auto &uv0 = prim->lines.add_attr<vec3f>("uv0");
            auto &uv1 = prim->lines.add_attr<vec3f>("uv1");

            parallel_for(prim->polys.size(), [&](size_t i) {
                auto [start, len] = prim->polys[i];
                int scanbase = linebase + scansum[i];
                for (int j = 0; j + 1 < len; j++) {
                    uv0[scanbase] = {uvs[loop_uv[start + j]][0], uvs[loop_uv[start + j]][1], 0};
                    uv1[scanbase] = {uvs[loop_uv[start + j + 1]][0], uvs[loop_uv[start + j + 1]][1], 0};
                    prim->lines[scanbase++] = vec2i(prim->loops[start + j], prim->loops[start + j + 1]);
                }
            });
        }
        prim->loops.clear();
        prim->polys.clear();
        prim->loops.erase_attr("uvs");
        prim->uvs.clear();
    }
}

struct PrimitiveLineify : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        if (get_param<bool>("from_poly")) {
            primLineify(prim.get(), get_param<bool>("with_uv"));
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveLineify, {/* inputs: */
                              {{"primitive", "prim"}},
                              /* outputs: */
                              {
                                  {"primitive", "prim"},
                              },
                              /* params: */
                              {
                                  {"bool", "from_poly", "1"},
                                  {"bool", "with_uv", "1"},
                              },
                              /* category: */
                              {
                                  "primitive",
                              }});

struct PointsToZSParticles : INode {
    void apply() override {
        using namespace zs;

        // primitive
        auto inParticles = get_input<PrimitiveObject>("prim");
        bool include_customed_properties = get_input2<bool>("add_customed_attr");
        auto &obj = inParticles->attr<vec3f>("pos");

        auto outParticles = std::make_shared<ZenoParticles>();

        // primitive binding
        outParticles->prim = inParticles;

        /// category, size
        std::size_t size{obj.size()};

        outParticles->category = ZenoParticles::mpm;

        // per vertex (node) vol, pos, vel
        auto ompExec = zs::omp_exec();

        // particles
        outParticles->sprayedOffset = obj.size();

        // attributes
        std::vector<zs::PropertyTag> tags{{"x", 3}, {"v", 3}};

        // prim attrib tags
        std::vector<zs::PropertyTag> auxVertAttribs{};
        for (auto &&[key, arr] : inParticles->verts.attrs) {
            const auto checkDuplication = [&tags](const std::string &name) {
                for (std::size_t i = 0; i != tags.size(); ++i)
                    if (tags[i].name == name.data())
                        return true;
                return false;
            };
            if (checkDuplication(key) || key == "pos" || key == "vel")
                continue;
            const auto &k{key};
            match(
                [&k, &auxVertAttribs](const std::vector<vec3f> &vals) {
                    auxVertAttribs.push_back(PropertyTag{k, 3});
                },
                [&k, &auxVertAttribs](const std::vector<float> &vals) {
                    auxVertAttribs.push_back(PropertyTag{k, 1});
                },
                [&k, &auxVertAttribs](const std::vector<vec3i> &vals) {},
                [&k, &auxVertAttribs](const std::vector<int> &vals) {},
                [](...) { throw std::runtime_error("what the heck is this type of attribute!"); })(arr);
        }
        tags.insert(std::end(tags), std::begin(auxVertAttribs), std::end(auxVertAttribs));

        fmt::print("[PointsToZSParticles] to be converted vert properties:\n");
        for (auto tag : tags)
            fmt::print("vert prop tag: [{}, {}]\n", tag.name, tag.numChannels);

        outParticles->particles = std::make_shared<typename ZenoParticles::particles_t>(tags, size, memsrc_e::host);
        auto &pars = outParticles->getParticles(); // tilevector
        {
            ompExec(zs::range(size), [pars = proxy<execspace_e::openmp>({}, pars), &obj, &inParticles,
                                      &auxVertAttribs](size_t pi) mutable {
                // pos
                pars.tuple(dim_c<3>, "x", pi) = obj[pi];

                // vel
                if (inParticles->has_attr("vel"))
                    pars.tuple(dim_c<3>, "v", pi) = inParticles->attr<vec3f>("vel")[pi];
                else
                    pars.tuple(dim_c<3>, "v", pi) = zs::vec<float, 3>::zeros();

                // additional attributes
                for (auto &prop : auxVertAttribs) {
                    if (prop.numChannels == 3)
                        pars.tuple(dim_c<3>, prop.name, pi) = inParticles->attr<vec3f>(std::string{prop.name})[pi];
                    else
                        pars(prop.name, pi) = inParticles->attr<float>(std::string{prop.name})[pi];
                }
            });

            pars = pars.clone({memsrc_e::device});
        }

        set_output("ZSParticles", outParticles);
    }
};

ZENDEFNODE(PointsToZSParticles, {
                                    {"prim", {"bool", "add_customed_attr", "1"}},
                                    {"ZSParticles"},
                                    {},
                                    {"conversion"},
                                });

struct PointsToZSParticles2 : INode {
    void apply() override {
        using namespace zs;

        // primitive
        auto inParticles = get_input<PrimitiveObject>("prim");
        bool include_customed_properties = get_input2<bool>("add_customed_attr");
        auto selected_props_list = get_input2<std::string>("selected_custom_attr_tags");
        std::set<std::string> selected_props;
        auto processTags = [](std::string tags, std::set<std::string> &res) {
            using Ti = RM_CVREF_T(std::string::npos);
            Ti st = tags.find_first_not_of(", ", 0);
            for (auto ed = tags.find_first_of(", ", st + 1); ed != std::string::npos;
                 ed = tags.find_first_of(", ", st + 1)) {
                res.insert(tags.substr(st, ed - st));
                // fmt::print("extract [{}, {}): [{}]\n", st, ed, res.back());
                st = tags.find_first_not_of(", ", ed);
                if (st == std::string::npos)
                    break;
            }
            if (st != std::string::npos && st < tags.size()) {
                res.insert(tags.substr(st));
                // fmt::print("extract [{}, npos): [{}]\n", st, res.back());
            }
        };
        processTags(selected_props_list, selected_props);
        auto &obj = inParticles->attr<vec3f>("pos");

        auto outParticles = std::make_shared<ZenoParticles>();

        // primitive binding
        outParticles->prim = inParticles;

        /// category, size
        std::size_t size{obj.size()};

        outParticles->category = ZenoParticles::mpm;

        // per vertex (node) vol, pos, vel
        auto ompExec = zs::omp_exec();

        // particles
        outParticles->sprayedOffset = obj.size();

        // attributes
        std::vector<zs::PropertyTag> tags{{"x", 3}, {"v", 3}};

        // prim attrib tags
        std::vector<zs::PropertyTag> auxVertAttribs{};
        if (include_customed_properties) {
            for (auto &&[key, arr] : inParticles->verts.attrs) {
                const auto checkDuplication = [&tags](const std::string &name) {
                    for (std::size_t i = 0; i != tags.size(); ++i)
                        if (tags[i].name == name.data())
                            return true;
                    return false;
                };
                if (checkDuplication(key) || key == "pos" || key == "vel")
                    continue;
                if (selected_props.size() > 0 && selected_props.find(key) == selected_props.end())
                    continue;
                const auto &k{key};
                match(
                    [&k, &auxVertAttribs](const std::vector<vec3f> &vals) {
                        auxVertAttribs.push_back(PropertyTag{k, 3});
                    },
                    [&k, &auxVertAttribs](const std::vector<float> &vals) {
                        auxVertAttribs.push_back(PropertyTag{k, 1});
                    },
                    [&k, &auxVertAttribs](const std::vector<vec3i> &vals) {},
                    [&k, &auxVertAttribs](const std::vector<int> &vals) {},
                    [](...) { throw std::runtime_error("what the heck is this type of attribute!"); })(arr);
            }
        }
        tags.insert(std::end(tags), std::begin(auxVertAttribs), std::end(auxVertAttribs));

        fmt::print("[PointsToZSParticles] to be converted vert properties:\n");
        for (auto tag : tags)
            fmt::print("vert prop tag: [{}, {}]\n", tag.name, tag.numChannels);

        outParticles->particles = std::make_shared<typename ZenoParticles::particles_t>(tags, size, memsrc_e::host);
        auto &pars = outParticles->getParticles(); // tilevector
        {
            ompExec(zs::range(size), [pars = proxy<execspace_e::openmp>({}, pars), &obj, &inParticles,
                                      &auxVertAttribs](size_t pi) mutable {
                // pos
                pars.tuple(dim_c<3>, "x", pi) = obj[pi];

                // vel
                if (inParticles->has_attr("vel"))
                    pars.tuple(dim_c<3>, "v", pi) = inParticles->attr<vec3f>("vel")[pi];
                else
                    pars.tuple(dim_c<3>, "v", pi) = zs::vec<float, 3>::zeros();

                // additional attributes
                for (auto &prop : auxVertAttribs) {
                    if (prop.numChannels == 3)
                        pars.tuple(dim_c<3>, prop.name, pi) = inParticles->attr<vec3f>(std::string{prop.name})[pi];
                    else
                        pars(prop.name, pi) = inParticles->attr<float>(std::string{prop.name})[pi];
                }
            });

            pars = pars.clone({memsrc_e::device});
        }

        set_output("ZSParticles", outParticles);
    }
};

ZENDEFNODE(PointsToZSParticles2, {
                                     {
                                         "prim",
                                         {"bool", "add_customed_attr", "1"},
                                         {"string", "selected_custom_attr_tags", ""},
                                     },
                                     {"ZSParticles"},
                                     {},
                                     {"conversion"},
                                 });

} // namespace zeno