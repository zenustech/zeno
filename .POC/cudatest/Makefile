x:
	clear
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
	make -Cbuild
	build/OpenFDB

d:
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
	make -Cbuild
	gdb build/OpenFDB -ex r
