x:
	cmake -Bbuild
	make -Cbuild -j12
	build/zeno.main

d:
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
	make -Cbuild -j12
	gdb build/zeno.main -ex r
