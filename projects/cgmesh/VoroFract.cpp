#include <vector>
#include <voro++/voro++.hh>
#include <tuple>

struct PolyMesh {
    std::vector<std::tuple<float, float, float>> vert;
    std::vector<std::tuple<int, int>> poly;
    std::vector<int> loop;
};

template <class MeshT>
void vorosplit(std::vector<MeshT> &outMeshes) {
	int nx, ny, nz;
	double x, y, z;
    voro::voronoicell_neighbor c;
    std::vector<int> neigh, f_vert;
    std::vector<double> v;

    voro::pre_container pcon(-3,3,-3,3,0,6,false,false,false);
	pcon.import("pack_six_cube");
	pcon.guess_optimal(nx,ny,nz);

    voro::container con(-3,3,-3,3,0,6,nx,ny,nz,false,false,false,8);
	pcon.setup(con);

    voro::c_loop_all cl(con);
	if(cl.start()) do if(con.compute_cell(c,cl)) {
		cl.pos(x, y, z);

		c.neighbors(neigh);
		c.face_vertices(f_vert);
		c.vertices(x, y, z, v);

        auto &mesh = outMeshes.emplace_back();

        for (int i = 0; i < (int)v.size(); i += 3) {
            mesh.vert.emplace_back(v[i], v[i+1], v[i+2]);
        }

		for(int i = 0, j = 0; i < (int)neigh.size(); i++) {
            int jbeg = j + 1;
            int jend = jbeg + f_vert[j];
            for (j = jbeg; j < jend; j++) {
                mesh.loop.push_back(f_vert[j]);
            }
            mesh.poly.emplace_back(jbeg, jend - jbeg);
        }

	} while (cl.inc());
}
