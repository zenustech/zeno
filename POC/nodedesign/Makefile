x:
	cmake -Bbuild
	make -Cbuild -j8
	build/ZenoEditor

d:
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
	make -Cbuild -j8
	gdb build/ZenoEditor -ex r
