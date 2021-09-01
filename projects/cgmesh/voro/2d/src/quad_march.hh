#ifndef QUAD_MARCH_HH
#define QUAD_MARCH_HH

#include "ctr_quad_2d.hh"

namespace voro {

template<int ca>
class quad_march {
	public:
		const int ma;
		int s;
		int ns;
		int p;
		quadtree* list[32];
		quad_march(quadtree *q);
		void step();
		inline quadtree* cu() {
			return list[p];
		}
	private:
		inline quadtree* up(quadtree* q) {
			return ca<2?(ca==0?q->qne:q->qnw):(ca==2?q->qnw:q->qsw);
		}
		inline quadtree* down(quadtree* q) {
			return ca<2?(ca==0?q->qse:q->qsw):(ca==2?q->qne:q->qse);
		}
};

}

#endif
