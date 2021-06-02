z:
	./dist.sh
	du -h /tmp/dist
	docker run -v /tmp/dist:/tmp/dist -it algebr/openface

x:
	cmake -B build
	make -C build -j `python -c 'from multiprocessing import cpu_count; print(cpu_count() * 2)'`
	USE_GDB= ZEN_OPEN=assets/mesh.zsg ./run.sh

y:
	scripts/alldlls.sh
