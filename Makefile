run: all
	build/main/zeno.main

debug: all
	gdb build/main/zeno.main -ex r

test: all
	build/tests/zeno.tests

all:
	cmake -Bbuild
	make -Cbuild -j12

config:
	cmake -Bbuild
	ccmake -Bbuild
