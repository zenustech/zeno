#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

#include "kernel/tiled_vector_ops.hpp"
#include "zensim/container/Bcht.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "kernel/topology.hpp"

namespace zeno {

struct BuildSurfaceHalfEdgeStructure : zeno::INode {
	using T = float;

	virtual void apply() override {
		using namespace zs;
		using vec2i = zs::vec<int, 2>;
		using vec3i = zs::vec<int, 3>;

		auto zsparticles = get_input<ZenoParticles>("zsparticles");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
                throw std::runtime_error("the input zsparticles has no surface tris");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
                throw std::runtime_error("the input zsparticles has no surface lines");
            if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag))
                throw std::runtime_error("the input zsparticles has no surface lines");

			auto& tris = (*zsparticles)[ZenoParticles::s_surfTriTag];
			auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
			auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];
			
			auto& halfEdge = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
			halfEdge = typename ZenoParticles::particles_t({{"to_vertex",1},{"face",1},{"edge",1},{"opposite_he",1},{"next_he",1}},
				tris.size() * 3,zs::memsrc_e::device,0);

			auto cudaPol = zs::cuda_exec();

#if 0

			constexpr auto space = zs::execspace_e::cuda;

			TILEVEC_OPS::fill(cudaPol,halfEdge,"to_vertex",reinterpret_bits<T>((int)-1));
			TILEVEC_OPS::fill(cudaPol,halfEdge,"face",reinterpret_bits<T>((int)-1));
			TILEVEC_OPS::fill(cudaPol,halfEdge,"edge",reinterpret_bits<T>((int)-1));
			TILEVEC_OPS::fill(cudaPol,halfEdge,"opposite_he",reinterpret_bits<T>((int)-1));
			TILEVEC_OPS::fill(cudaPol,halfEdge,"next_he",reinterpret_bits<T>((int)-1));

			// we might also need a space hash structure here, map from [i1,i2]->[ej]
			bcht<vec2i,int,true,universal_hash<vec2i>,32> de2fi{halfEdge.get_allocator(),halfEdge.size()};

			cudaPol(zs::range(tris.size()), [
				tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi),halfEdge = proxy<space>({},halfEdge)] ZS_LAMBDA(int ti) mutable {
					auto fe_inds = tris.pack(dim_c<3>,"fe_inds",ti).reinterpret_bits(int_c);
					auto tri = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);

					vec3i nos{};
					for(int i = 0;i != 3;++i) {
						if(auto no = de2fi.insert(vec2i{tri[i],tri[(i+1) % 3]});no >= 0){
							nos[i] = no;
							halfEdge("to_vertex",no) = reinterpret_bits<T>(tri[i]);
							halfEdge("face",no) = reinterpret_bits<T>(ti);
							halfEdge("edge",no) = reinterpret_bits<T>(fe_inds[i]);
							// halfEdge("next_he",no) = ti * 3 + (i+1) % 3;
						} else {
							// some error happen

						}						
					}
					for(int i = 0;i != 3;++i)
						halfEdge("next_he",nos[i]) = reinterpret_bits<T>(nos[(i+1) % 3]);
			});
			cudaPol(zs::range(halfEdge.size()),
				[halfEdge = proxy<space>({},halfEdge),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int hei) mutable {
					auto idx0 = reinterpret_bits<int>(halfEdge("to_vertex",hei));
					auto nexthei = reinterpret_bits<int>(halfEdge("next_he",hei));
					auto idx1 = reinterpret_bits<int>(halfEdge("to_vertex",nexthei));
					if(auto no = de2fi.query(vec2i{idx1,idx0});no >= 0)
						halfEdge("opposite_he",hei) = reinterpret_bits<T>(no);
					else	
						halfEdge("opposite_he",hei) = reinterpret_bits<T>((int)-1);
			});

			points.append_channels(cudaPol,{{"he_inds",1}});
			lines.append_channels(cudaPol,{{"he_inds",1}});
			tris.append_channels(cudaPol,{{"he_inds",1}});

			cudaPol(zs::range(lines.size()),[
				lines = proxy<space>({},lines),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int li) mutable {
					auto linds = lines.pack(dim_c<2>,"ep_inds",li).reinterpret_bits(int_c);
					if(auto no = de2fi.query(vec2i{linds[0],linds[1]});no >= 0){
						lines("he_inds",li) = reinterpret_bits<T>((int)no);
					}else {
						// some algorithm bug
					}
			});

			cudaPol(zs::range(tris.size()),[
				points = proxy<space>({},points),tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int ti) mutable {
					auto tinds = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);
					if(auto no = de2fi.query(vec2i{tinds[0],tinds[1]});no >= 0){
						tris("he_inds",ti) = reinterpret_bits<T>((int)no);
					}else {
						// some algorithm bug
					}

					for(int i = 0;i != 3;++i) {
						if(auto no = de2fi.query(vec2i{tinds[i],tinds[(i+1) % 3]});no >= 0){
							points("he_inds",tinds[i]) = reinterpret_bits<T>((int)no);
						}else {
							// some algorithm bug
						}						
					}
			});
#else
			if(build_surf_half_edge(cudaPol,tris,lines,points,halfEdge))
				fmt::print(fg(fmt::color::red),"fail building surf half edge\n");
#endif

			set_output("zsparticles",zsparticles);
			// zsparticles->setMeta("de2fi",std::move())
	}

};


ZENDEFNODE(BuildSurfaceHalfEdgeStructure, {{{"zsparticles"}},
							{{"zsparticles"}},
							{},
							{"ZSGeometry"}});


// visualize the one-ring points, lines, and tris
struct VisualizeOneRingNeighbors : zeno::INode {
	using T = float;
	virtual void apply() override {
		using namespace zs;
		auto zsparticles = get_input<ZenoParticles>("zsparticles");
		constexpr int MAX_NEIGHS = 8;

		if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
			throw std::runtime_error("the input zsparticles has no surface tris");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
			throw std::runtime_error("the input zsparticles has no surface lines");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag))
			throw std::runtime_error("the input zsparticles has no surface lines");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfHalfEdgeTag))
			throw std::runtime_error("the input zsparticles has no half edges");

		const auto& verts = zsparticles->getParticles();
		const auto& tris = (*zsparticles)[ZenoParticles::s_surfTriTag];
		const auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
		const auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];
		const auto& half_edges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];


		auto cudaPol = zs::cuda_exec();
		constexpr auto space = zs::execspace_e::cuda;

		auto one_ring_points = typename ZenoParticles::particles_t({{"x",3},{"active",1}},points.size() * (MAX_NEIGHS + 1),zs::memsrc_e::device,0);
		TILEVEC_OPS::fill(cudaPol,one_ring_points,"active",(T)0);

		// auto one_ring_lines = typename ZenoParticles::particles_t({{"x",3},{"active",1}},points.size() * (MAX_NEIGHS + 1));
		// auto one_ring_tris = typename ZenoParticles::particles_t({{"x",3},{"active",1}},points.size() * (MAX_NEIGHS + 1));

		cudaPol(zs::range(points.size()),[
				verts = proxy<space>({},verts),
				one_ring_points = proxy<space>({},one_ring_points),
				// one_ring_lines = proxy<space>({},one_ring_lines),
				// one_ring_tris = proxy<space>({},one_ring_tris),
				points = proxy<space>({},points),
				lines = proxy<space>({},lines),
				tris = proxy<space>({},tris),
				half_edges = proxy<space>({},half_edges)] ZS_LAMBDA(int pi) mutable {
			// calculate one-ring neighbored points
			one_ring_points("active",pi * MAX_NEIGHS + 0) = (T)1.0;
			auto pidx = reinterpret_bits<int>(points("inds",pi));
			one_ring_points.tuple(dim_c<3>,"x",pi * MAX_NEIGHS + 0) = verts.pack(dim_c<3>,"x",pidx);

			auto he_idx = reinterpret_bits<int>(points("he_inds",pi));

			zs::vec<int,MAX_NEIGHS> pneighs = get_one_ring_neigh_points<MAX_NEIGHS>(he_idx,half_edges);
			for(int i = 0;i != MAX_NEIGHS;++i){
				if(pneighs[i] < 0)
					break;
				auto npidx = reinterpret_bits<int>(points("inds",pneighs[i]));
				one_ring_points("active",pi * MAX_NEIGHS + i + 1) = (T)1.0;
				one_ring_points.tuple(dim_c<3>,"x",pi * MAX_NEIGHS + i + 1) = verts.pack(dim_c<3>,"x",npidx);
			}

		});

		one_ring_points = one_ring_points.clone({zs::memsrc_e::host});
		auto pn_prim = std::make_shared<zeno::PrimitiveObject>();
		auto& pn_verts = pn_prim->verts;
		auto& pn_lines = pn_prim->lines;

		pn_verts.resize(points.size() * (MAX_NEIGHS + 1));
		pn_lines.resize(points.size() * MAX_NEIGHS);
		constexpr auto omp_space = execspace_e::openmp;
		auto ompPol = omp_exec();    

		ompPol(zs::range(points.size()),
			[one_ring_points = proxy<omp_space>({},one_ring_points),&pn_verts,&pn_lines] (int pi) {
				for(int i = 0;i != MAX_NEIGHS + 1;++i)
					pn_verts[pi * (MAX_NEIGHS + 1) + i] = one_ring_points.pack(dim_c<3>,"x",pi * (MAX_NEIGHS + 1) + i).to_array();
				for(int i = 0;i != MAX_NEIGHS;++i)
					pn_lines[pi * MAX_NEIGHS + i] = zeno::vec2i(pi * (MAX_NEIGHS + 1) + 0,pi * (MAX_NEIGHS + 1) + i + 1);
		});  

		set_output("prim",std::move(pn_prim));
	}
};


ZENDEFNODE(VisualizeOneRingNeighbors, {{{"zsparticles"}},
							{{"prim"}},
							{},
							{"ZSGeometry"}});


};