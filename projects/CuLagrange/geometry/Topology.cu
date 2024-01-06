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

struct ZSMarkIsland : zeno::INode {
	using vec2i = zs::vec<int,2>;
	using vec3i = zs::vec<int,3>;
	using vec4i = zs::vec<int,4>;
	using OrderedEdge = zs::vec<int,6,2>;
	using T = float;

	virtual void apply() override {
		using namespace zs;

		auto cudaPol = zs::cuda_exec();
		constexpr auto space = zs::execspace_e::cuda;

		auto zsparticles = get_input<ZenoParticles>("zsparticles");
		auto& elms = zsparticles->getQuadraturePoints();
		auto& verts = zsparticles->getParticles();

		zs::Vector<vec2i> edge_topos{verts.get_allocator(),0};
		
		auto simplex_size = elms.getPropertySize("inds");
		int nm_edges_per_elm = 0;
		if(simplex_size == 2)
			nm_edges_per_elm = 1;
		else if(simplex_size == 3)
			nm_edges_per_elm = 3;
		else if(simplex_size == 4)
			nm_edges_per_elm = 6;

		edge_topos.resize(elms.size() * nm_edges_per_elm);
		OrderedEdge ordered_edges{
			0,1,
			1,2,
			2,0,
			0,3,
			1,3,
			2,3
		};

		cudaPol(zs::range(elms.size()),[
			elms = proxy<space>({},elms),
			edge_topos = proxy<space>(edge_topos),
			nm_edges_per_elm = nm_edges_per_elm,
			ordered_edges,
			simplex_size = simplex_size] ZS_LAMBDA(int ei) mutable {
				for(int i = 0;i != nm_edges_per_elm;++i) {
					auto a = reinterpret_bits<int>(elms("inds",ordered_edges[i][0],ei));
					auto b = reinterpret_bits<int>(elms("inds",ordered_edges[i][1],ei));
					edge_topos[ei * nm_edges_per_elm + i] = vec2i{a,b};
				}
		});

		zs::Vector<bool> topo_disable_buffer{verts.get_allocator(),verts.size()};
		zs::Vector<int> island_buffer{verts.get_allocator(),verts.size()};

		mark_disconnected_island(cudaPol,edge_topos,island_buffer);

		auto markTag = get_param<std::string>("mark_tag");
		if(!verts.hasProperty(markTag))
			verts.append_channels(cudaPol,{{markTag,1}});
		
		cudaPol(zs::range(verts.size()),[
			verts = proxy<space>({},verts),
			markTag = zs::SmallString(markTag),
			island_buffer = proxy<space>(island_buffer)] ZS_LAMBDA(int vi) mutable {
				verts(markTag,vi) = reinterpret_bits<T>((int)island_buffer[vi]);
				verts(markTag,vi) = (T)island_buffer[vi];
		});

		set_output("zsparticles",zsparticles);
	}
};

ZENDEFNODE(ZSMarkIsland, {{{"zsparticles"}},
							{
								{"zsparticles"}
							},
							{
								{"string","mark_tag","mark_tag"}
							},
							{"ZSGeometry"}});

struct ZSManifoldCheck : zeno::INode {
	virtual void apply() override {
		using namespace zs;

		auto cudaPol = zs::cuda_exec();
		constexpr auto space = zs::execspace_e::cuda;

		auto zsparticles = get_input<ZenoParticles>("zsparticles");
		const auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();

		// auto is_manifold = is_manifold_check(cudaPol,tris);

		auto ret = std::make_shared<zeno::NumericObject>();
        ret->set<bool>(is_manifold_check(cudaPol,tris));
        set_output("is_manifold", std::move(ret));
	}
};

ZENDEFNODE(ZSManifoldCheck, {{{"zsparticles"}},
							{
								{"is_manifold"}
							},
							{
								// {"string","mark_tag","mark_tag"}
							},
							{"ZSGeometry"}});

struct DoTopogicalColoring : zeno::INode {
	virtual void apply() override {
			using namespace zs;
			using vec2i = zs::vec<int, 2>;
			using vec3i = zs::vec<int, 3>;
			using vec4i = zs::vec<int, 4>;

			auto cudaPol = zs::cuda_exec();
			constexpr auto space = zs::execspace_e::cuda;

			auto zsparticles = get_input<ZenoParticles>("zsparticles");
			// const auto& verts = zsparticles->getParticles();
			auto& elms = zsparticles->getQuadraturePoints();
			// auto& tris = (*zsparticles)[ZenoParticles::s_surfTriTag];
			// const auto& tets = zsparticles->getQuadraturePoints();
			auto cdim = elms.getPropertySize("inds");
			auto color_tag = get_param<std::string>("colorTag");

			auto do_sort_color = get_param<bool>("sort_color");

			if(!elms.hasProperty(color_tag))
				elms.append_channels(cudaPol,{{color_tag,1}});

			zs::Vector<vec4i> topos{elms.get_allocator(),elms.size()};
			cudaPol(zs::range(elms.size()),[
				elms = proxy<space>({},elms),
				cdim,
				topos = proxy<space>(topos)] ZS_LAMBDA(int ti) mutable {
					topos[ti] = vec4i::uniform(-1);
					for(int i = 0;i != cdim;++i) {
						topos[ti][i] = zs::reinterpret_bits<int>(elms("inds",i,ti));
					}
			});

			zs::Vector<float> colors{elms.get_allocator(),elms.size()};
			// std::cout << "do topological coloring" << std::endl;
			topological_coloring(cudaPol,topos,colors);
			zs::Vector<int> reordered_map{elms.get_allocator(),elms.size()};
			cudaPol(zs::range(reordered_map.size()),[reordered_map = proxy<space>(reordered_map)] ZS_LAMBDA(int ti) mutable {reordered_map[ti] = ti;});
			// if(do_sort_color)
			zs::Vector<int> color_offset{elms.get_allocator(),0};
			sort_topology_by_coloring_tag(cudaPol,colors,reordered_map,color_offset);

			// std::cout << "finish topological coloring" << std::endl;

			cudaPol(zs::range(elms.size()),[
				elms = proxy<space>({},elms),
				color_tag = zs::SmallString(color_tag),
				topos = proxy<space>(topos),
				reordered_map = proxy<space>(reordered_map),
				cdim,
				colors = proxy<space>(colors)] ZS_LAMBDA(int ei) mutable {
					elms(color_tag,ei) = colors[reordered_map[ei]];
					for(int i = 0;i != cdim;++i)
						elms("inds",i,ei) = zs::reinterpret_bits<float>(topos[reordered_map[ei]][i]);
			});

			zsparticles->setMeta("color_offset",color_offset);
			printf("offset : ");
			for(int i = 0;i != color_offset.size();++i)
				printf("%d\t",color_offset.getVal(i));
			printf("\n");
			set_output("zsparticles",zsparticles);
	}
};

ZENDEFNODE(DoTopogicalColoring, {{{"zsparticles"}},
							{
								{"zsparticles"}
							},
							{
								{"string","colorTag","colorTag"},
								// {"bool","sort_color","1"}
							},
							{"ZSGeometry"}});

};