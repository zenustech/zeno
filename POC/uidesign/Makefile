x:
	cmake -Bbuild
	make -Cbuild -j
	build/ZenoEditor

d:
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
	make -Cbuild -j
	gdb build/ZenoEditor -ex r
