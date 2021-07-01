O=arts/rigid3.zsg

default: dist

dist: all
	make -C build install
	./dist.sh

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/tmp-install
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`

debug_all:
	cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/tmp/tmp-install
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`

configure:
	cmake -B build
	ccmake -B build

test: all
	build/tests/zentest

run: all
	ZEN_OPEN=$O ./run.sh

debug: debug_all
	USE_GDB=1 ZEN_SPROC=1 ZEN_OPEN=$O ./run.sh

.PHONY: all debug_all debug run prepare install wheel default
