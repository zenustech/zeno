#include "TopoUtils.hpp"
#include "zensim/container/Bht.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"

namespace zeno {

void compute_surface_neighbors(zs::CudaExecutionPolicy &pol, ZenoParticles::particles_t &sfs,
                               ZenoParticles::particles_t &ses, ZenoParticles::particles_t &svs) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using vec2i = zs::vec<int, 2>;
    using vec3i = zs::vec<int, 3>;
    sfs.append_channels(pol, {{"ff_inds", 3}, {"fe_inds", 3}, {"fp_inds", 3}});
    ses.append_channels(pol, {{"fe_inds", 2}});

    fmt::print("sfs size: {}, ses size: {}, svs size: {}\n", sfs.size(), ses.size(), svs.size());

    bht<int, 2, int> etab{sfs.get_allocator(), sfs.size() * 3};
    etab.reset(pol, true);
    Vector<int> sfi{sfs.get_allocator(), sfs.size() * 3}; // surftri indices corresponding to edges in the table
    /// @brief compute ff neighbors
    {
        pol(range(sfs.size()), [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs),
                                sfi = proxy<space>(sfi)] __device__(int ti) mutable {
            auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
            for (int i = 0; i != 3; ++i)
                if (auto no = etab.insert(vec2i{tri[i], tri[(i + 1) % 3]}); no >= 0) {
                    sfi[no] = ti;
                } else {
                    auto oti = sfi[etab.query(vec2i{tri[i], tri[(i + 1) % 3]})];
                    auto otri = sfs.pack(dim_c<3>, "inds", oti).reinterpret_bits(int_c);
                    printf("the same directed edge <%d, %d> has been inserted twice! original sfi %d <%d, %d, %d>, cur "
                           "%d <%d, %d, %d>\n",
                           tri[i], tri[(i + 1) % 3], oti, otri[0], otri[1], otri[2], ti, tri[0], tri[1], tri[2]);
                }
        });
        pol(range(sfs.size()), [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs),
                                sfi = proxy<space>(sfi)] __device__(int ti) mutable {
            auto neighborIds = vec3i::uniform(-1);
            auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
            for (int i = 0; i != 3; ++i)
                if (auto no = etab.query(vec2i{tri[(i + 1) % 3], tri[i]}); no >= 0) {
                    neighborIds[i] = sfi[no];
                }
            sfs.tuple(dim_c<3>, "ff_inds", ti) = neighborIds.reinterpret_bits(float_c);
            sfs.tuple(dim_c<3>, "fe_inds", ti) = vec3i::uniform(-1); // default initialization
        });
    }
    /// @brief compute fe neighbors
    {
        auto sfindsOffset = sfs.getPropertyOffset("inds");
        auto sfFeIndsOffset = sfs.getPropertyOffset("fe_inds");
        auto seFeIndsOffset = ses.getPropertyOffset("fe_inds");
        pol(range(ses.size()),
            [etab = proxy<space>(etab), sfs = proxy<space>({}, sfs), ses = proxy<space>({}, ses),
             sfi = proxy<space>(sfi), sfindsOffset, sfFeIndsOffset, seFeIndsOffset] __device__(int li) mutable {
                auto findLineIdInTri = [](const auto &tri, int v0, int v1) -> int {
                    for (int loc = 0; loc < 3; ++loc)
                        if (tri[loc] == v0 && tri[(loc + 1) % 3] == v1)
                            return loc;
                    return -1;
                };
                auto neighborTris = vec2i::uniform(-1);
                auto line = ses.pack(dim_c<2>, "inds", li).reinterpret_bits(int_c);

                {
                    if (auto no = etab.query(line); no >= 0) {
                        // tri
                        auto triNo = sfi[no];
                        auto tri = sfs.pack(dim_c<3>, sfindsOffset, triNo).reinterpret_bits(int_c);
                        auto loc = findLineIdInTri(tri, line[0], line[1]);
                        if (loc == -1) {
                            printf("ridiculous, this edge <%d, %d> does not belong to tri <%d, %d, %d>\n", line[0],
                                   line[1], tri[0], tri[1], tri[2]);
                        } else {
                            sfs(sfFeIndsOffset + loc, triNo) = reinterpret_bits<float>(li);
                            // edge
                            neighborTris[0] = triNo;
                        }
                    }
                }
                vec2i rline{line[1], line[0]};
                {
                    if (auto no = etab.query(rline); no >= 0) {
                        // tri
                        auto triNo = sfi[no];
                        auto tri = sfs.pack(dim_c<3>, sfindsOffset, triNo).reinterpret_bits(int_c);
                        auto loc = findLineIdInTri(tri, rline[0], rline[1]);
                        if (loc == -1) {
                            printf("ridiculous, this edge <%d, %d> does not belong to tri <%d, %d, %d>\n", rline[0],
                                   rline[1], tri[0], tri[1], tri[2]);
                        } else {
                            sfs(sfFeIndsOffset + loc, triNo) = reinterpret_bits<float>(li);
                            // edge
                            neighborTris[1] = triNo;
                        }
                    }
                }
                ses.tuple(dim_c<2>, seFeIndsOffset, li) = neighborTris.reinterpret_bits(float_c);
            });
    }
    /// @brief compute fp neighbors
    /// @note  surface vertex index is not necessarily consecutive, thus hashing
    {
        bht<int, 1, int, 32> vtab{svs.get_allocator(), svs.size()};
        Vector<int> svi{etab.get_allocator(), svs.size()}; // surftri indices corresponding to edges in the table
        // svs
        pol(range(svs.size()), [vtab = proxy<space>(vtab), svs = proxy<space>({}, svs),
                                svi = proxy<space>(svi)] __device__(int vi) mutable {
            int vert = reinterpret_bits<int>(svs("inds", vi));
            if (auto no = vtab.insert(vert); no >= 0)
                svi[no] = vi;
        });
        //
        pol(range(sfs.size()), [vtab = proxy<space>(vtab), sfs = proxy<space>({}, sfs),
                                svi = proxy<space>(svi)] __device__(int ti) mutable {
            auto neighborIds = vec3i::uniform(-1);
            auto tri = sfs.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
            for (int i = 0; i != 3; ++i)
                if (auto no = vtab.query(tri[i]); no >= 0) {
                    neighborIds[i] = svi[no];
                }
            sfs.tuple(dim_c<3>, "fp_inds", ti) = neighborIds.reinterpret_bits(float_c);
        });
    }
}

void update_surface_cell_normals(zs::CudaExecutionPolicy &pol, ZenoParticles::particles_t &verts,
                                 const zs::SmallString &xTag, std::size_t vOffset, ZenoParticles::particles_t &tris,
                                 const zs::SmallString &triNrmTag, ZenoParticles::particles_t &lines,
                                 const zs::SmallString &biNrmTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;

    if (!verts.hasProperty(xTag))
        throw std::runtime_error(fmt::format("missing property [{}] for vertex positions.", xTag));
    if (!tris.hasProperty("inds"))
        throw std::runtime_error("missing property [inds] for surface triangles.");
    if (!lines.hasProperty("fe_inds") || !lines.hasProperty("inds"))
        throw std::runtime_error("missing property [fe_inds]/[inds] for surface edges.");
    if (!tris.hasProperty(triNrmTag))
        throw std::runtime_error(fmt::format("missing property [{}] for surface triangles.", triNrmTag));
    if (!lines.hasProperty(biNrmTag))
        throw std::runtime_error(fmt::format("missing property [{}] for surface edges.", biNrmTag));

    pol(range(tris.size()), [verts = proxy<space>(verts), xOffset = verts.getPropertyOffset(xTag), vOffset = vOffset,
                             tris = proxy<space>({}, tris), triNrmTag] ZS_LAMBDA(int ti) mutable {
        auto tri = tris.pack(dim_c<3>, "inds", ti).reinterpret_bits(int_c);
        auto t0 = verts.pack(dim_c<3>, xOffset, tri[0] + vOffset);
        auto t1 = verts.pack(dim_c<3>, xOffset, tri[1] + vOffset);
        auto t2 = verts.pack(dim_c<3>, xOffset, tri[2] + vOffset);
        using vec3 = RM_CVREF_T(t0);
        using T = typename vec3::value_type;
        auto nrm = (t1 - t0).cross(t2 - t0);
        if (auto len = nrm.l2NormSqr(); len > detail::deduce_numeric_epsilon<T>() * 10)
            nrm /= zs::sqrt(len);
        else
            nrm = vec3::zeros();
        tris.tuple(dim_c<3>, triNrmTag, ti) = nrm;
    });

    pol(range(lines.size()), [verts = proxy<space>(verts), xOffset = verts.getPropertyOffset(xTag), vOffset = vOffset,
                              tris = proxy<space>({}, tris), lines = proxy<space>({}, lines), triNrmTag,
                              biNrmTag] ZS_LAMBDA(int ei) mutable {
        auto fe_inds = lines.pack(dim_c<2>, "fe_inds", ei).reinterpret_bits(int_c);
        auto ne = zs::vec<float, 3>::zeros();
        if (fe_inds[0] >= 0)
            ne += tris.pack(dim_c<3>, triNrmTag, fe_inds[0]);
        if (fe_inds[1] >= 0)
            ne += tris.pack(dim_c<3>, triNrmTag, fe_inds[1]);
        ne /= ne.length();

        // be careful when extarcting vertex positions
        auto e_inds = lines.pack(dim_c<2>, "inds", ei).reinterpret_bits(int_c) + vOffset;
        auto e0 = verts.pack(dim_c<3>, xOffset, e_inds[0]);
        auto e1 = verts.pack(dim_c<3>, xOffset, e_inds[1]);
        auto e10 = e1 - e0;

        using vec3 = RM_CVREF_T(e0);
        using T = typename vec3::value_type;
        auto nrm = ne.cross(e10);
        if (auto len = nrm.l2NormSqr(); len > detail::deduce_numeric_epsilon<T>() * 10)
            nrm /= zs::sqrt(len);
        else
            nrm = vec3::zeros();
        lines.tuple(dim_c<3>, biNrmTag, ei) = nrm;
    });
}

} // namespace zeno