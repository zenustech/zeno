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

			// zsparticles->setMeta("de2fi",std::move())
	}

};

};