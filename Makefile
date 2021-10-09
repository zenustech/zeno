run:
	cmake -Bbuild
	make -Cbuild -j12 zeno.main
	build/main/zeno.main

debug:
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
	make -Cbuild -j12 zeno.main
	gdb build/main/zeno.main -ex r

test:
	cmake -Bbuild
	make -Cbuild -j12 zeno.tests
	build/tests/zeno.tests

all:
	cmake -Bbuild
	make -Cbuild -j12

config:
	cmake -Bbuild
	ccmake -Bbuild
