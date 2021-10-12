run:
	cmake -B/tmp/zeno-build
	make -C/tmp/zeno-build -j12 zs.main
	/tmp/zeno-build/src/zs/main/zs.main

debug:
	cmake -B/tmp/zeno-build -DCMAKE_BUILD_TYPE=Debug
	make -C/tmp/zeno-build -j12 zs.main
	gdb /tmp/zeno-build/src/zs/main/zs.main -ex r

test:
	cmake -B/tmp/zeno-build
	make -C/tmp/zeno-build -j12 zs.tests
	/tmp/zeno-build/src/zs/tests/zs.tests

all:
	cmake -B/tmp/zeno-build
	make -C/tmp/zeno-build -j12

config:
	cmake -B/tmp/zeno-build
	ccmake -B/tmp/zeno-build
