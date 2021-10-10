run:
	cmake -Bbuild
	make -Cbuild -j12 zs.main
	build/src/zs/main/zs.main

debug:
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
	make -Cbuild -j12 zs.main
	gdb build/src/zs/main/zs.main -ex r

test:
	cmake -Bbuild
	make -Cbuild -j12 zs.tests
	build/src/zs/tests/zs.tests

all:
	cmake -Bbuild
	make -Cbuild -j12

config:
	cmake -Bbuild
	ccmake -Bbuild
