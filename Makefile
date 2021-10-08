x:
	cmake -Bbuild
	make -Cbuild -j12
	build/bin/ZenoEditor

d:
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
	make -Cbuild -j12
	gdb build/bin/ZenoEditor -ex r
