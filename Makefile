O=assets/prim.zsg

default: build run

install: copy_libs
	python/setup.py install

wheel_pkg: copy_libs
	python/setup.py bdist_wheel

copy_libs: build
	make -C build install
	cp -d /tmp/tmp-install/lib/*.so* zen/lib/

build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/tmp-install
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`

debug_build:
	cmake -B build -DCMAKE_BUILD_TYPE=Debug
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`

run: build
	ZEN_OPEN=$O ./run.sh

no_build_run:
	ZEN_OPEN=$O ./run.sh

debug: debug_build
	USE_GDB=1 ZEN_SPROC=1 ZEN_OPEN=$O ./run.sh
