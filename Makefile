run: zs_editor
	/tmp/zeno-build/editor/zs_editor

debug: zs_editor
	gdb /tmp/zeno-build/editor/zs_editor -ex r

test: zs_tests
	/tmp/zeno-build/tests/zs_tests

%:
	test -d /tmp/zeno-ccache || mkdir /tmp/zeno-ccache
	test -d ~/.cache/ccache || ln -s /tmp/zeno-ccache ~/.cache/ccache
	test -d /tmp/zeno-build || mkdir /tmp/zeno-build
	test -d build || ln -s /tmp/zeno-build build
	cmake -B /tmp/zeno-build -G Ninja
	cmake --build /tmp/zeno-build --parallel

config:
	ccmake -B /tmp/zeno-build
