#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <random>
#include <stdint.h>

#include "./rxmesh_static.h"
#include "./geodesic_kernel.cuh"

namespace zeno::rxmesh {


    struct RXMeshNode : zeno::INode {
        public:
        void geodesic_rxmesh(rxmesh::RXMeshStatic*        rxmesh,
                     const std::vector<uint32_t>& h_seeds,
                     const std::vector<uint32_t>& h_limits,
                     const std::vector<std::vector<uint32_t>>& toplesets,
                     const std::vector<std::vector<float>>& coords,
                     std::vector<float>& clrs) {
            using namespace rxmesh;
            constexpr uint32_t blockThreads = 256;

            // coords
            auto d_coords = rxmesh->add_vertex_attribute(coords, "coords");

            // toplesets
            auto d_toplesets = rxmesh->add_vertex_attribute(toplesets, "topleset");


            // RXMesh launch box
            rxmesh::LaunchBox<blockThreads> launch_box;
            rxmesh->prepare_launch_box({rxmesh::Op::VV},
                                    launch_box,
                                    (void*)relax_ptp_rxmesh<blockThreads>,
                                    true);

            // Geodesic distance attribute for all vertices 
            auto rxmesh_geo = rxmesh->add_vertex_attribute<float>("geo", 1u);
            rxmesh_geo->reset(std::numeric_limits<float>::infinity(), rxmesh::HOST);
            rxmesh->for_each_vertex(rxmesh::HOST, [&](const ElementHandle vh) {
                uint32_t v_id = rxmesh->map_to_global_v(vh);
                for (uint32_t s : h_seeds) {
                    if (s == v_id) {
                        (*rxmesh_geo)(vh) = 0;
                        break;
                    }
                }
            });
            rxmesh_geo->move(rxmesh::HOST, rxmesh::DEVICE);

            // second buffer for geodesic distance for double buffering
            auto rxmesh_geo_2 =
                rxmesh->add_vertex_attribute<float>("geo2", 1u, rxmesh::DEVICE);

            rxmesh_geo_2->copy_from(*rxmesh_geo, rxmesh::DEVICE, rxmesh::DEVICE);


            // Error
            uint32_t *d_error(nullptr);
            uint32_t h_error = 0;
            CUDA_ERROR(cudaMalloc((void**)&d_error, sizeof(uint32_t)));

            // double buffer
            rxmesh::VertexAttribute<float>* double_buffer[2] = {rxmesh_geo.get(),
                                                    rxmesh_geo_2.get()};

            // actual computation
            uint32_t d = 0;
            uint32_t i(1), j(2);
            uint32_t iter     = 0;
            uint32_t max_iter = 2 * h_limits.size();
            while (i < j && iter < max_iter) {
                ++iter;
                if (i < (j / 2)) {
                    i = j / 2;
                }

                CUDA_ERROR(cudaDeviceSynchronize());
                CUDA_ERROR(cudaGetLastError());

                // compute new geodesic
                rxmesh::relax_ptp_rxmesh<blockThreads>
                    <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                        rxmesh->get_context(),
                        *d_coords,
                        *double_buffer[!d],
                        *double_buffer[d],
                        *d_toplesets,
                        i,
                        j,
                        d_error,
                        std::numeric_limits<float>::infinity(),
                        float(1e-3));

                CUDA_ERROR(cudaMemcpy(
                    &h_error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                CUDA_ERROR(cudaMemset(d_error, 0, sizeof(uint32_t)));

                const uint32_t n_cond = h_limits[i + 1] - h_limits[i];

                if (n_cond == h_error) {
                    i++;
                }
                if (j < coords.size() - 1) {
                    j++;
                }

                d = !d;
            }
            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());

            rxmesh_geo->copy_from(*double_buffer[d], rxmesh::DEVICE, rxmesh::HOST);

            rxmesh->for_each_vertex(rxmesh::HOST, [&](const ElementHandle vh) {
                uint32_t v_id = rxmesh->map_to_global_v(vh);
                clrs[v_id] = (*rxmesh_geo)(vh);
            });

            GPU_FREE(d_error);
        }
            
        virtual void apply() override {
            auto prim = get_input<PrimitiveObject>("prim");

            std::vector<std::vector<uint32_t>> fv;
            fv.resize(prim->tris.size());
            for (auto i = 0; i < prim->tris.size(); ++i) {
                fv[i].emplace_back(prim->tris[i][0]);
                fv[i].emplace_back(prim->tris[i][1]);
                fv[i].emplace_back(prim->tris[i][2]);
            }
            // prim->verts
            std::vector<std::vector<float>> coords;
            coords.resize(prim->verts.size());
            for (auto i = 0; i < prim->verts.size(); ++i) {
                coords[i].emplace_back(prim->verts[i][0]);
                coords[i].emplace_back(prim->verts[i][1]);
                coords[i].emplace_back(prim->verts[i][2]);
            }

            zeno::rxmesh::RXMeshStatic* rxmesh = new zeno::rxmesh::RXMeshStatic(fv);
            
            std::vector<std::vector<uint32_t>> vv;
            vv.resize(coords.size());
            for (auto & i : fv) {
                for (int j = 0; j < i.size(); ++j) {
                    if (j == 0) {
                        vv[i[i.size() - 1]].emplace_back(i[0]);
                        vv[i[0]].emplace_back(i[i.size() - 1]);
                    } else {
                        vv[i[j - 1]].emplace_back(i[j]);
                        vv[i[j]].emplace_back(i[j - 1]);
                    }
                }
            }

            std::vector<uint32_t> sorted_index;
            std::vector<uint32_t> limits;
            std::vector<std::vector<uint32_t>> toplesets;

            sorted_index.clear();
            sorted_index.resize(prim->verts.size());
            limits.clear();
            limits.resize(prim->verts.size() / 2);
            uint32_t level = 0;
            uint32_t p     = 0;

            toplesets.clear();
            toplesets.resize(prim->verts.size());
            for (int i = 0 ; i < prim->verts.size(); ++i) {
                toplesets[i].emplace_back(INVALID32);
            }

            // Generate Seeds
            std::vector<uint32_t> h_seeds(1);
            std::random_device    dev;
            std::mt19937          rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist(
                0, prim->verts.size());
            for (auto& s : h_seeds) {
                s = dist(rng);
                sorted_index[p] = s;
                p++;
                if (toplesets[s][0] == INVALID32) {
                    toplesets[s][0] = level;
                }
            }

            limits.push_back(0);
            for (uint32_t i = 0; i < p; i++) {
                const uint32_t v = sorted_index[i];
                if (toplesets[v][0] > level) {
                    level++;
                    limits.push_back(i);
                }
                for (auto j : vv[v]) {
                    if (toplesets[j][0] == INVALID32) {
                        toplesets[j][0] = toplesets[v][0] + 1;
                        sorted_index[p] = j;
                        p++;
                    }
                }
            }

            std::vector<float> clrs;
            clrs.resize(prim->verts.size(), std::numeric_limits<float>::infinity());
            // RXMesh Impl
            geodesic_rxmesh(rxmesh, h_seeds, limits, toplesets, coords, clrs);

            float maxf = 0;
            for (auto i = 0; i < clrs.size(); ++i) {
                assert(clrs[i] > 0);
                maxf = std::max(maxf, clrs[i]);
            }
            maxf += 0.001;

            auto &clr = prim->verts.add_attr<zeno::vec3f>("clr");
            for (auto i = 0; i < prim->verts.size(); ++i) {
                vec3f c(clrs[i] / maxf * 60, clrs[i] / maxf * 80, clrs[i] / maxf * 100);
                clr[i] = c;
                clr[i] = zeno::vec3f(0.2f * clrs[i] / maxf, 0.8f * clrs[i] / maxf, 0.9f * clrs[i] / maxf);
            }
            set_output("prim", std::move(prim));
        }
    };
    ZENDEFNODE(RXMeshNode, {
         { // inputs:
            {"PrimitiveObject", "prim"},
         },
         { // outputs:
            {"PrimitiveObject", "prim"}
         },
         { // params

         },
         { // category
             "RXMeshNode"
         },
    });


} // namespace zeno

