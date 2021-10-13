run: zs_editor
	/tmp/zeno-build/editor/zs_editor

debug: zs_editor
	gdb /tmp/zeno-build/editor/zs_editor -ex r

test: zs_tests
	/tmp/zeno-build/tests/zs_tests

all:
	cmake -B/tmp/zeno-build
	make -C/tmp/zeno-build -j12

config:
	cmake -B/tmp/zeno-build
	ccmake -B/tmp/zeno-build

%:
	test -d /tmp/zeno-build || mkdir /tmp/zeno-build
	test -d build || ln -s /tmp/zeno-build build
	cmake -B/tmp/zeno-build
	make -C/tmp/zeno-build -j12 $<
