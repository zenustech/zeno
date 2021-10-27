T=nvptx64-nvidia-cuda

x:
	/opt/sycl/bin/clang++ -Wl,-rpath=/opt/sycl/lib -std=c++17 -fsycl -fsycl-targets=$T main.cpp -o a.out
	./a.out
