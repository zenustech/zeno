#include "quad_march.hh"

namespace voro {

template<int ca>
quad_march<ca>::quad_march(quadtree *q) : ma(1<<30), s(0), p(0) {
	list[p]=q;
	while(list[p]->id==NULL) {list[p+1]=up(list[p]);p++;}
	ns=ma>>p;
}

template<int ca>
void quad_march<ca>::step() {
	if(ns>=ma) {s=ma+1;return;}
	while(down(list[p-1])==list[p]) p--;
	list[p]=down(list[p-1]);
	while(list[p]->id==NULL) {list[p+1]=up(list[p]);p++;}
	s=ns;ns+=ma>>p;
}

// Explicit instantiation
template class quad_march<0>;
template class quad_march<1>;
template class quad_march<2>;
template class quad_march<3>;

}
