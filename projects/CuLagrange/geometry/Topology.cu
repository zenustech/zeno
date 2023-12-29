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
		constexpr auto space = zs::execspace_e::cuda;

		auto zsparticles = get_input<ZenoParticles>("zsparticles");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag) && zsparticles->category == ZenoParticles::category_e::tet)
			throw std::runtime_error("the input tet zsparticles has no surface tris");
		// if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
		//     throw std::runtime_error("the input zsparticles has no surface lines");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag) && zsparticles->category == ZenoParticles::category_e::tet)
			throw std::runtime_error("the input tet zsparticles has no surface points");

		auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();
		// auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
		auto& points = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfVertTag] : zsparticles->getParticles();
		
		auto& halfEdge = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];
		halfEdge = typename ZenoParticles::particles_t({{"local_vertex_id",1},{"to_face",1},{"opposite_he",1},{"next_he",1}},
			tris.size() * 3,zs::memsrc_e::device,0);

		auto cudaPol = zs::cuda_exec();

		points.append_channels(cudaPol,{{"he_inds",1}});
		// lines.append_channels(cudaPol,{{"he_inds",1}});
		tris.append_channels(cudaPol,{{"he_inds",1}});

#if 0

		// constexpr auto space = zs::execspace_e::cuda;

		TILEVEC_OPS::fill(cudaPol,halfEdge,"to_vertex",reinterpret_bits<T>((int)-1));
		TILEVEC_OPS::fill(cudaPol,halfEdge,"to_face",reinterpret_bits<T>((int)-1));
		TILEVEC_OPS::fill(cudaPol,halfEdge,"to_edge",reinterpret_bits<T>((int)-1));
		TILEVEC_OPS::fill(cudaPol,halfEdge,"opposite_he",reinterpret_bits<T>((int)-1));
		TILEVEC_OPS::fill(cudaPol,halfEdge,"next_he",reinterpret_bits<T>((int)-1));

		// we might also need a space hash structure here, map from [i1,i2]->[ej]
		bcht<vec2i,int,true,universal_hash<vec2i>,32> de2fi{halfEdge.get_allocator(),halfEdge.size()};

		cudaPol(zs::range(tris.size()), [
			tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi),halfEdge = proxy<space>({},halfEdge)] ZS_LAMBDA(int ti) mutable {
				auto fe_inds = tris.pack(dim_c<3>,"fe_inds",ti).reinterpret_bits(int_c);
				auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);

				vec3i nos{};
				for(int i = 0;i != 3;++i) {
					if(auto no = de2fi.insert(vec2i{tri[i],tri[(i+1) % 3]});no >= 0){
						nos[i] = no;
						halfEdge("to_vertex",no) = reinterpret_bits<T>(tri[i]);
						halfEdge("to_face",no) = reinterpret_bits<T>(ti);
						halfEdge("to_edge",no) = reinterpret_bits<T>(fe_inds[i]);
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
				auto linds = lines.pack(dim_c<2>,"inds",li).reinterpret_bits(int_c);
				if(auto no = de2fi.query(vec2i{linds[0],linds[1]});no >= 0){
					lines("he_inds",li) = reinterpret_bits<T>((int)no);
				}else {
					// some algorithm bug
				}
		});

		if(!tris.hasProperty("fp_inds") || tris.getPropertySize("fp_inds") != 3) {
			throw std::runtime_error("the tris has no fp_inds channel");
		}

		cudaPol(zs::range(tris.size()),[
			tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int ti) mutable {

				auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
				if(auto no = de2fi.query(vec2i{tri[0],tri[1]});no >= 0){
					tris("he_inds",ti) = reinterpret_bits<T>((int)no);
				}else {
					// some algorithm bug
					printf("could not find half edge : %d %d\n",tri[0],tri[1]);
				}
				// auto tinds = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);
				// for(int i = 0;i != 3;++i) {
				// 	if(auto no = de2fi.query(vec2i{tri[i],tri[(i+1) % 3]});no >= 0){
				// 		points("he_inds",tinds[i]) = reinterpret_bits<T>((int)no);
				// 	}else {
				// 		// some algorithm bug
				// 		printf("could not find half edge : %d %d\n",tri[i],tri[(i+1) % 3]);
				// 	}						
				// }
		});

		cudaPol(zs::range(tris.size()),[
			points = proxy<space>({},points),tris = proxy<space>({},tris),de2fi = proxy<space>(de2fi)] ZS_LAMBDA(int ti) mutable {
				auto tinds = tris.pack(dim_c<3>,"fp_inds",ti).reinterpret_bits(int_c);
				auto tri = tris.pack(dim_c<3>,"inds",ti).reinterpret_bits(int_c);
				for(int i = 0;i != 3;++i) {
					if(auto no = de2fi.query(vec2i{tri[i],tri[(i+1) % 3]});no >= 0){
						points("he_inds",tinds[i]) = reinterpret_bits<T>((int)no);
					}else {
						// some algorithm bug
						printf("could not find half edge : %d %d\n",tri[i],tri[(i+1) % 3]);
					}						
				}
		});
#else
		auto accept_non_manifold = get_input2<bool>("accept_non_manifold");
		if(!accept_non_manifold) {
			if(!build_surf_half_edge(cudaPol,tris,points,halfEdge))
				throw std::runtime_error("fail building surf half edge");
		}else {
			if(!build_surf_half_edge_robust(cudaPol,tris,halfEdge))
				throw std::runtime_error("fail building surf half edge");
		}
#endif

		zs::bht<int,1,int> edgeSet{tris.get_allocator(),tris.size() * 3};	
		zs::bht<int,1,int> boundaryHalfEdgeSet{tris.get_allocator(),tris.size() * 3};
		edgeSet.reset(cudaPol,true);
		boundaryHalfEdgeSet.reset(cudaPol,true);
		cudaPol(zs::range(halfEdge.size()),[
			halfedges = proxy<space>({},halfEdge),
			boundaryHalfEdgeSet = proxy<space>(boundaryHalfEdgeSet),
			edgeSet = proxy<space>(edgeSet),
			tris = proxy<space>({},tris)] ZS_LAMBDA(int hi) mutable {
				auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
				if(ti < 0) {
					printf("oops!!! halfedge with no incident triangle!!!\n");
					return;
				}
					
				auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
				auto local_idx = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
				zs::vec<int,2> edge{tri[local_idx],tri[(local_idx + 1) % 3]};

				auto ohi = zs::reinterpret_bits<int>(halfedges("opposite_he",hi));
				if(ohi < 0)
					boundaryHalfEdgeSet.insert(hi);
				if(ohi >= 0 && edge[0] > edge[1])
					return;

				edgeSet.insert(hi);
		});

		auto &surfEdges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
		surfEdges = typename ZenoParticles::particles_t({{"inds", 2},{"he_inds",1}}, edgeSet.size(),zs::memsrc_e::device,0);
		cudaPol(zip(zs::range(edgeSet.size()),edgeSet._activeKeys),[
			halfedges = proxy<space>({},halfEdge),
			surfEdges = proxy<space>({},surfEdges),
			tris = proxy<space>({},tris)] ZS_LAMBDA(auto ei,const auto& hi_vec) mutable {
				auto hi = hi_vec[0];
				auto ti = zs::reinterpret_bits<int>(halfedges("to_face",hi));
				if(ti < 0) {
					printf("oops!!! halfedge with no incident triangle!!!\n");
					return;
				}

				auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
				auto local_idx = zs::reinterpret_bits<int>(halfedges("local_vertex_id",hi));
				zs::vec<int,2> edge{tri[local_idx],tri[(local_idx + 1) % 3]};	
				
				surfEdges.tuple(dim_c<2>,"inds",ei) = edge.reinterpret_bits(float_c);
				surfEdges("he_inds",ei) = reinterpret_bits<float>((int)hi);
		});

		auto& boundaryHalfEdges = (*zsparticles)[ZenoParticles::s_surfBoundaryEdgeTag];
		boundaryHalfEdges = typename ZenoParticles::particles_t({{"he_inds",1}},
			boundaryHalfEdgeSet.size(),zs::memsrc_e::device,0);

		cudaPol(zip(zs::range(boundaryHalfEdgeSet.size()),boundaryHalfEdgeSet._activeKeys),[
			boundaryHalfEdges = boundaryHalfEdges.begin("he_inds",dim_c<1>,int_c)] ZS_LAMBDA(int id,const auto& key) mutable {
				boundaryHalfEdges[id] = key[0];
		});

		set_output("zsparticles",zsparticles);
	}

};


ZENDEFNODE(BuildSurfaceHalfEdgeStructure, {{
								{"zsparticles"},
								{"bool","accept_non_manifold","0"},
							},
							{{"zsparticles"}},
							{},
							{"ZSGeometry"}});

struct BuildTetrahedraHalfFacet : zeno::INode {
	using T = float;
	virtual void apply() override {
		using namespace zs;
		auto cudaPol = zs::cuda_exec();


		auto zsparticles = get_input<ZenoParticles>("zsparticles");

		auto& tets = zsparticles->getQuadraturePoints();
		tets.append_channels(cudaPol,{{"hf_inds",1}});

		auto& halfFacet = (*zsparticles)[ZenoParticles::s_tetHalfFacetTag];
		halfFacet = typename ZenoParticles::particles_t({{"opposite_hf",1},{"next_hf",1},{"to_tet",1},{"local_idx",1}},
				tets.size() * 4,zs::memsrc_e::device,0);

		build_tetrahedra_half_facet(cudaPol,tets,halfFacet);

		set_output("zsparticles",zsparticles);
	}
};

ZENDEFNODE(BuildTetrahedraHalfFacet, {{{"zsparticles"}},
							{{"zsparticles"}},
							{},
							{"ZSGeometry"}});

struct BuildSurfaceLinesStructure : zeno::INode {
	using T = float;
	virtual void apply() override {
		using namespace zs;
		using vec2i = zs::vec<int, 2>;
		using vec3i = zs::vec<int, 3>;
		auto cudaPol = zs::cuda_exec();
		constexpr auto space = zs::execspace_e::cuda;

		auto zsparticles = get_input<ZenoParticles>("zsparticles");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag) && zsparticles->category == ZenoParticles::category_e::tet)
			throw std::runtime_error("the input tet zsparticles has no surface tris");

		auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? (*zsparticles)[ZenoParticles::s_surfTriTag] : zsparticles->getQuadraturePoints();
		zs::bht<int,2,int> edgeSet{tris.get_allocator(),tris.size() * 3};
		edgeSet.reset(cudaPol,true);

		cudaPol(zs::range(tris.size()),[
			tris = proxy<space>({},tris),
			edgeSet = proxy<space>(edgeSet)] ZS_LAMBDA(int ti) mutable {
				auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
				for(int i = 0;i != 3;++i) {
					auto idx0 = tri[(i + 0) % 3];
					auto idx1 = tri[(i + 1) % 3];
					if(idx0 < idx1)
						edgeSet.insert(vec2i{idx0,idx1});
				}
		});
		
		auto &surfEdges = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
		surfEdges = typename ZenoParticles::particles_t({{"inds", 2}}, edgeSet.size(),zs::memsrc_e::device,0);	
		cudaPol(zip(zs::range(edgeSet.size()),edgeSet._activeKeys),[
			surfEdges = proxy<space>({},surfEdges)] ZS_LAMBDA(auto ei,const auto& pair) mutable {
				surfEdges.tuple(dim_c<2>,"inds",ei) = pair.reinterpret_bits(float_c);
		});	

		set_output("zsparticles",zsparticles);
	}
};

ZENDEFNODE(BuildSurfaceLinesStructure, {{{"zsparticles"}},
							{{"zsparticles"}},
							{},
							{"ZSGeometry"}});

struct VisualTetrahedraHalfFacet : zeno::INode {
	using T = float;
	// using dtiles_t = zs::TileVector<T,32>;
	virtual void apply() override {
		using namespace zs;
		auto cudaPol = zs::cuda_exec();
		constexpr auto space = zs::execspace_e::cuda;

		auto zsparticles = get_input<ZenoParticles>("zsparticles");
		const auto& tets = zsparticles->getQuadraturePoints();
		const auto& verts = zsparticles->getParticles();
		const auto& halfFacet = (*zsparticles)[ZenoParticles::s_tetHalfFacetTag];
		
		auto tet_centers = typename ZenoParticles::particles_t({{"x",3}},tets.size(),zs::memsrc_e::device,0);
		auto neighbor_centers = typename ZenoParticles::particles_t({{"x",3}},tets.size() * 4,zs::memsrc_e::device,0);

		cudaPol(zs::range(tets.size()),[
			tets = proxy<space>({},tets),
			verts = proxy<space>({},verts),
			tet_centers = proxy<space>({},tet_centers)] ZS_LAMBDA(int ti) mutable {
				// tet_centers.tuple(dim_c<3>,"x",ti) = zs::vec<T,3>::zeros();
				auto tet_center = zs::vec<T,3>::zeros();
				auto tet = tets.pack(dim_c<4>,"inds",ti,int_c);
				for(int i = 0;i != 4;++i)
					tet_center += verts.pack(dim_c<3>,"x",tet[i]) / (T)4.0;
				tet_centers.tuple(dim_c<3>,"x",ti) = tet_center;
		});

		cudaPol(zs::range(tets.size()),[
			tets = proxy<space>({},tets),
			verts = proxy<space>({},verts),
			tet_centers = proxy<space>({},tet_centers),
			neighbor_centers = proxy<space>({},neighbor_centers),
			halfFacet = proxy<space>({},halfFacet)] ZS_LAMBDA(int ti) mutable {
				auto hf_idx = zs::reinterpret_bits<int>(tets("hf_inds",ti));
				if(hf_idx < 0) {
					printf("invalid hf_idx : %d\n",hf_idx);
					return;
				}
				auto tet = tets.pack(dim_c<3>,"inds",ti,int_c);
				// zs::vec<T,3> ncenters[4] = {};
				for(int i = 0;i != 4;++i) {
					auto opposite_hf_idx = zs::reinterpret_bits<int>(halfFacet("opposite_hf",hf_idx));
					if(opposite_hf_idx >= 0) {
						if(opposite_hf_idx >= halfFacet.size()) {
							printf("opposite_hf_idx = %d exceeding size of halfFacet : %d\n", (int)opposite_hf_idx, (int)halfFacet.size());
							return;
						}
						auto nti = zs::reinterpret_bits<int>(halfFacet("to_tet",opposite_hf_idx));
						if(nti >= tet_centers.size() || nti < 0) {
							printf("invalid nti : %d\n",nti);
							return;
						}
						neighbor_centers.tuple(dim_c<3>,"x",ti * 4 + i) = tet_centers.pack(dim_c<3>,"x",nti);
					}else {
						auto tcenter = zs::vec<T,3>::zeros();
						for(int j = 0;j != 3;++j) {
							tcenter += verts.pack(dim_c<3>,"x",tet[(j + 1) % 3]) /  (T)3.0;
						}
						neighbor_centers.tuple(dim_c<3>,"x",ti * 4 + i) = tcenter;
					}
					hf_idx = zs::reinterpret_bits<int>(halfFacet("next_hf",hf_idx));
				}
		});

		constexpr auto omp_space = execspace_e::openmp;
		auto ompPol = omp_exec();  
		tet_centers = tet_centers.clone({zs::memsrc_e::host});
		neighbor_centers = neighbor_centers.clone({zs::memsrc_e::host});

		auto tf_vis = std::make_shared<zeno::PrimitiveObject>();
		auto& tf_verts = tf_vis->verts;
		auto& tf_lines = tf_vis->lines;
		tf_verts.resize(tets.size() * 5);
		tf_lines.resize(tets.size() * 4);

		ompPol(zs::range(tets.size()),[
			tet_centers = proxy<omp_space>({},tet_centers),
			neighbor_centers = proxy<omp_space>({},neighbor_centers),
			&tf_verts,&tf_lines] (int ti) mutable {
				auto tc = tet_centers.pack(dim_c<3>,"x",ti);
				tf_verts[ti * 5 + 0] = tc.to_array();
				for(int i = 0;i != 4;++i) {
					auto ntc = neighbor_centers.pack(dim_c<3>,"x",ti * 4 + i);
					tf_verts[ti * 5 + i + 1] = ntc.to_array();
					tf_lines[ti * 4 + i] = zeno::vec2i(ti * 5 + 0,ti * 5 + i + 1);
				}
		});

		set_output("halfFacet_vis",std::move(tf_vis));
	}
};

ZENDEFNODE(VisualTetrahedraHalfFacet, {{{"zsparticles"}},
							{
								{"halfFacet_vis"}
							},
							{},
							{"ZSGeometry"}});


#define MAX_NEIGHS 32

// visualize the one-ring points, lines, and tris
struct VisualizeOneRingNeighbors : zeno::INode {
	using T = float;
	virtual void apply() override {
		using namespace zs;
		auto zsparticles = get_input<ZenoParticles>("zsparticles");
		// constexpr int MAX_NEIGHS = 8;

		// if(!zsparticles->hasAuxData(ZenoParticles::s_surfTriTag))
		// 	throw std::runtime_error("the input zsparticles has no surface tris");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfEdgeTag))
			throw std::runtime_error("the input zsparticles has no surface lines");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfVertTag))
			throw std::runtime_error("the input zsparticles has no surface lines");
		if(!zsparticles->hasAuxData(ZenoParticles::s_surfHalfEdgeTag))
			throw std::runtime_error("the input zsparticles has no half edges");

		const auto& verts = zsparticles->getParticles();
        auto& tris = zsparticles->category == ZenoParticles::category_e::tet ? 
            (*zsparticles)[ZenoParticles::s_surfTriTag] : 
            zsparticles->getQuadraturePoints();
		const auto& lines = (*zsparticles)[ZenoParticles::s_surfEdgeTag];
		const auto& points = (*zsparticles)[ZenoParticles::s_surfVertTag];
		const auto& half_edges = (*zsparticles)[ZenoParticles::s_surfHalfEdgeTag];


		// std::cout << "size of half edge : " << half_edges.size() << std::endl;
		// std::cout << "size of verts : " << verts.size() << std::endl;
		// std::cout << "size of lines : " << lines.size() << std::endl;
		// std::cout << "size of points : " << points.size() << std::endl;
		// std::cout << "size of tris : " << tris.size() << std::endl;

		auto cudaPol = zs::cuda_exec();
		constexpr auto space = zs::execspace_e::cuda;

		auto one_ring_points = typename ZenoParticles::particles_t({{"x",3},{"active",1}},points.size() * (MAX_NEIGHS + 1),zs::memsrc_e::device,0);
		TILEVEC_OPS::fill(cudaPol,one_ring_points,"active",(T)0);

		auto one_ring_lines = typename ZenoParticles::particles_t({{"x",3},{"active",1}},points.size() * (MAX_NEIGHS * 2),zs::memsrc_e::device,0);
		TILEVEC_OPS::fill(cudaPol,one_ring_lines,"active",(T)0);

		// auto one_ring_tris = typename ZenoParticles::particles_t({{"x",3},{"active",1}},points.size() * (MAX_NEIGHS + 1),zs::memsrc_e::device,0);


		// auto one_ring_lines = typename ZenoParticles::particles_t({{"x",3},{"active",1}},points.size() * (MAX_NEIGHS + 1));
		// auto one_ring_tris = typename ZenoParticles::particles_t({{"x",3},{"active",1}},points.size() * (MAX_NEIGHS + 1));

		// cudaPol(zs::range(lines.size()),
		// 	[lines = proxy<space>({},lines)] ZS_LAMBDA(int li) {
		// 		auto linds = lines.pack(dim_c<2>,"inds",li).reinterpret_bits(int_c);
		// 		printf("linds[%d] : %d %d\n",li,linds[0],linds[1]);
		// });

		// cudaPol(zs::range(half_edges.size()),
		// 	[half_edges = proxy<space>({},half_edges)] ZS_LAMBDA(int hei) {
		// 		auto id0 = reinterpret_bits<int>(half_edges("to_vertex",hei));
		// 		auto nhei = get_next_half_edge(hei,half_edges,1,false);
		// 		auto id1 = reinterpret_bits<int>(half_edges("to_vertex",nhei));
		// 		auto rhei = reinterpret_bits<int>(half_edges("opposite_he",hei));
		// 		auto rid0 = reinterpret_bits<int>(half_edges("to_vertex",rhei));
		// 		auto nrhei = get_next_half_edge(rhei,half_edges,1,false);
		// 		auto rid1 = reinterpret_bits<int>(half_edges("to_vertex",nrhei));
		// 		printf("half_edge[%d] : %d %d \t <-> half_edge[%d] : %d %d\n",hei,id0,id1,rhei,rid0,rid1);
		// });



		cudaPol(zs::range(points.size()),[
				verts = proxy<space>({},verts),
				one_ring_points = proxy<space>({},one_ring_points),
				// one_ring_lines = proxy<space>({},one_ring_lines),
				// one_ring_tris = proxy<space>({},one_ring_tris),
				points = proxy<space>({},points),
				lines = proxy<space>({},lines),
				tris = proxy<space>({},tris),
				// MAX_NEIGHS_V = wrapv<MAX_NEIGHS>{},
				half_edges = proxy<space>({},half_edges)] ZS_LAMBDA(int pi) mutable {
			// calculate one-ring neighbored points
			// constexpr int MAX_NEIGHS = RM_CVREF_T(MAX_NEIGHS_V)::value;
			one_ring_points("active",pi * (MAX_NEIGHS+1) + 0) = (T)1.0;
			auto pidx = reinterpret_bits<int>(points("inds",pi));
			one_ring_points.tuple(dim_c<3>,"x",pi * (MAX_NEIGHS+1) + 0) = verts.pack(dim_c<3>,"x",pidx);

			auto he_idx = reinterpret_bits<int>(points("he_inds",pi));

			zs::vec<int,MAX_NEIGHS> pneighs = get_one_ring_neigh_points<MAX_NEIGHS>(he_idx,half_edges,tris);
			// printf("one_ring_neighbors[%d] : %d %d %d %d %d %d\n",(int)pi,
			// 	(int)pneighs[0],(int)pneighs[1],(int)pneighs[2],(int)pneighs[3],(int)pneighs[4],(int)pneighs[5]);
			for(int i = 0;i != MAX_NEIGHS;++i){
				if(pneighs[i] < 0)
					break;
				// auto npidx = reinterpret_bits<int>(points("inds",pneighs[i]));
				one_ring_points("active",pi * (MAX_NEIGHS+1) + i + 1) = (T)1.0;
				one_ring_points.tuple(dim_c<3>,"x",pi * (MAX_NEIGHS+1) + i + 1) = verts.pack(dim_c<3>,"x",pneighs[i]);
			}

		});

		// cudaPol(zs::range(points.size()),[
		// 		verts = proxy<space>({},verts),
		// 		one_ring_lines = proxy<space>({},one_ring_lines),
		// 		points = proxy<space>({},points),
		// 		lines = proxy<space>({},lines),
		// 		// MAX_NEIGHS_V = wrapv<MAX_NEIGHS>{},
		// 		half_edges = proxy<space>({},half_edges)] ZS_LAMBDA(int pi) mutable {
		// 			// constexpr int MAX_NEIGHS = RM_CVREF_T(MAX_NEIGHS_V)::value;
		// 			auto he_idx = reinterpret_bits<int>(points("he_inds",pi));
		// 			zs::vec<int,MAX_NEIGHS> pneighs = get_one_ring_neigh_edges<MAX_NEIGHS>(he_idx,half_edges);
		// 			// printf("one_ring_line_neighbors[%d] : %d %d %d %d %d %d\n",(int)pi,
		// 			// 	(int)pneighs[0],(int)pneighs[1],(int)pneighs[2],(int)pneighs[3],(int)pneighs[4],(int)pneighs[5]);
		// 			for(int i = 0;i != MAX_NEIGHS;++i) {
		// 				if(pneighs[i] < 0)
		// 					break;
		// 				one_ring_lines("active",pi * (2 * MAX_NEIGHS) + 2 * i + 0) = (T)1.0;
		// 				one_ring_lines("active",pi * (2 * MAX_NEIGHS) + 2 * i + 1) = (T)1.0;
		// 				auto ne = lines.pack(dim_c<2>,"inds",pneighs[i]).reinterpret_bits(int_c);
		// 				one_ring_lines.tuple(dim_c<3>,"x",pi * (2 * MAX_NEIGHS) + 2 * i + 0) = verts.pack(dim_c<3>,"x",ne[0]);
		// 				one_ring_lines.tuple(dim_c<3>,"x",pi * (2 * MAX_NEIGHS) + 2 * i + 1) = verts.pack(dim_c<3>,"x",ne[1]);
		// 			}
		// });

		one_ring_points = one_ring_points.clone({zs::memsrc_e::host});
		auto pn_prim = std::make_shared<zeno::PrimitiveObject>();
		auto& pn_verts = pn_prim->verts;
		auto& pn_lines = pn_prim->lines;

		pn_verts.resize(points.size() * (MAX_NEIGHS + 1));
		pn_lines.resize(points.size() * MAX_NEIGHS);
		constexpr auto omp_space = execspace_e::openmp;
		auto ompPol = omp_exec();    

		ompPol(zs::range(points.size()),
			[one_ring_points = proxy<omp_space>({},one_ring_points),&pn_verts,
					&pn_lines] (int pi) {
				// constexpr int MAX_NEIGHS = RM_CVREF_T(MAX_NEIGHS_V)::value;
				int nm_active = 0;
				for(int i = 0;i != MAX_NEIGHS + 1;++i) {
					if(one_ring_points("active",pi * (MAX_NEIGHS+1) + i) > 0)
						nm_active++;
					else
						break;
					pn_verts[pi * (MAX_NEIGHS+1) + i] = one_ring_points.pack(dim_c<3>,"x",pi * (MAX_NEIGHS+1) + i).to_array();
					// if(i > 0) {
					// 	auto diff = pn_verts[pi * (MAX_NEIGHS+1) + i] - pn_verts[pi * (MAX_NEIGHS+1) + 0];
					// 	pn_verts[pi * (MAX_NEIGHS+1) + i] = pn_verts[pi * (MAX_NEIGHS+1) + 0] + diff * 0.9;
					// }
				}
				for(int i = 0;i < nm_active-1;++i)
					pn_lines[pi * MAX_NEIGHS + i] = zeno::vec2i(pi * (MAX_NEIGHS + 1) + 0,pi * (MAX_NEIGHS + 1) + i + 1);
		});  

		// for(int i = 0;i != pn_lines.size();++i)
		// 	std::cout << "pn_lines[" << i << "] : " << pn_lines[i][0] << "\t" << pn_lines[i][1] << std::endl;

		set_output("pn_prim",std::move(pn_prim));


		// one_ring_lines = one_ring_lines.clone({zs::memsrc_e::host});
		// auto en_prim = std::make_shared<zeno::PrimitiveObject>();
		// auto& en_verts = en_prim->verts;
		// auto& en_lines = en_prim->lines;

		// en_verts.resize(points.size() * (MAX_NEIGHS * 2));
		// en_lines.resize(points.size() * MAX_NEIGHS);

		// ompPol(zs::range(points.size()),
		// 	[one_ring_lines = proxy<omp_space>({},one_ring_lines),&en_verts,&en_lines] (int pi) {
		// 		// constexpr int MAX_NEIGHS = RM_CVREF_T(MAX_NEIGHS_V)::value;
		// 		int nm_active = 0;
		// 		for(int i = 0;i != 2*MAX_NEIGHS;++i) {
		// 			if(one_ring_lines("active",pi * MAX_NEIGHS * 2 + i) > 0)
		// 				nm_active++;
		// 			else
		// 				 break;
		// 			en_verts[pi * MAX_NEIGHS * 2 + i] = one_ring_lines.pack(dim_c<3>,"x",pi * MAX_NEIGHS * 2 + i).to_array();
		// 		}
		// 		int nm_active_edges = nm_active / 2;
		// 		for(int i = 0;i != nm_active_edges;++i)
		// 			en_lines[pi * MAX_NEIGHS + i] = zeno::vec2i(pi * MAX_NEIGHS * 2 + i * 2 + 0,pi * MAX_NEIGHS * 2 + i * 2 + 1);
		// });

		// set_output("en_prim",std::move(en_prim));
	}
};


ZENDEFNODE(VisualizeOneRingNeighbors, {{{"zsparticles"}},
							{{"pn_prim"}
								// ,{"en_prim"}
							},
							{},
							{"ZSGeometry"}});



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

struct BuildSurfFacetTetraNeighboring : zeno::INode {
		virtual void apply() override {
			using namespace zs;
			using vec2i = zs::vec<int, 2>;
			using vec3i = zs::vec<int, 3>;

			auto cudaPol = zs::cuda_exec();
			constexpr auto space = zs::execspace_e::cuda;

			auto zsparticles = get_input<ZenoParticles>("zsparticles");
			const auto& verts = zsparticles->getParticles();
			auto& tris = (*zsparticles)[ZenoParticles::s_surfTriTag];
			const auto& tets = zsparticles->getQuadraturePoints();

#if 0

			zs::bht<int,3,int> tris_htab{tris.get_allocator(),tris.size()};
			tris_htab.reset(cudaPol,true);
			zs::Vector<int> tris_id{tris.get_allocator(),tris.size()};
			cudaPol(zs::range(tris.size()),[
				tris = proxy<space>({},tris),
				tris_id = proxy<space>(tris_id),
				tris_htab = proxy<space>(tris_htab)] ZS_LAMBDA(int ti) mutable {
					auto tri = tris.pack(dim_c<3>,"inds",ti,int_c);
					order_indices(tri);
					for(int i = 0;i != 2;++i)
						if(tri[i] >= tri[i + 1])
							printf("invalid ordered tri : %d %d %d\n",tri[0],tri[1],tri[2]);
					auto no = tris_htab.insert(tri);
						tris_id[no] = ti;
			});

			cudaPol(zs::range(tris_id),[] ZS_LAMBDA(int& ti) {
				if(ti < 0)
					printf("invalid ordered tri[%d]\n");
			});

			if(!tris.hasProperty("ft_inds")) {
				tris.append_channels(cudaPol,{{"ft_inds",1}});
			}
			TILEVEC_OPS::fill(cudaPol,tris,"ft_inds",zs::reinterpret_bits<float>((int)-1));

			cudaPol(zs::range(tets.size()),[
				tets = proxy<space>({},tets),
				tris = proxy<space>({},tris),
				tris_htab = proxy<space>(tris_htab),
				tris_id = proxy<space>(tris_id)] ZS_LAMBDA(int ei) mutable {
					auto tet = tets.pack(dim_c<4>,"inds",ei,int_c);
					order_indices(tet);
					int tri_id = -1;
					if(auto no = tris_htab.query(vec3i{tet[1],tet[2],tet[3]});no >= 0)
						tri_id = tris_id[no];
					if(auto no = tris_htab.query(vec3i{tet[0],tet[2],tet[3]});no >= 0)
						tri_id = tris_id[no];
					if(auto no = tris_htab.query(vec3i{tet[0],tet[1],tet[3]});no >= 0)
						tri_id = tris_id[no];
					if(auto no = tris_htab.query(vec3i{tet[0],tet[1],tet[2]});no >= 0)
						tri_id = tris_id[no];
					if(tri_id >= 0)
						tris("ft_inds",tri_id) = zs::reinterpret_bits<float>(ei);
			});

			cudaPol(zs::range(tris.size()),[
				tris = proxy<space>({},tris)] ZS_LAMBDA(int ti) mutable {
					auto ei = zs::reinterpret_bits<int>(tris("ft_inds",ti));
					if(ei < 0) {
						printf("dangling surface tri %d detected\n",ti);
					}
			});

#else

			if(!tris.hasProperty("ft_inds")) {
				tris.append_channels(cudaPol,{{"ft_inds",1}});
			}
			TILEVEC_OPS::fill(cudaPol,tris,"ft_inds",zs::reinterpret_bits<float>((int)-1));

			compute_ft_neigh_topo(cudaPol,verts,tris,tets,"ft_inds");

#endif

			set_output("zsparticles",zsparticles);
		}
};

ZENDEFNODE(BuildSurfFacetTetraNeighboring, {{{"zsparticles"}},
							{
								{"zsparticles"}
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