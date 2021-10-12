run:
	cmake -B/tmp/zeno-build
	make -C/tmp/zeno-build -j12 zs_main
	/tmp/zeno-build/src/zs/main/zs_main

debug:
	cmake -B/tmp/zeno-build -DCMAKE_BUILD_TYPE=Debug
	make -C/tmp/zeno-build -j12 zs_main
	gdb /tmp/zeno-build/src/zs/main/zs_main -ex r

test:
	cmake -B/tmp/zeno-build
	make -C/tmp/zeno-build -j12 zs_tests
	/tmp/zeno-build/src/zs/tests/zs_tests

all:
	cmake -B/tmp/zeno-build
	make -C/tmp/zeno-build -j12

config:
	cmake -B/tmp/zeno-build
	ccmake -B/tmp/zeno-build
