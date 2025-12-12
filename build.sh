cmake -B build_gpu -DFAISS_ENABLE_GPU=ON \
				   -DFAISS_ENABLE_PYTHON=OFF \
				   -DBUILD_SHARED_LIBS=ON \
				   -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
				   -DCMAKE_CUDA_FLAGS="-Xcompiler=-std=c++17 --expt-relaxed-constexpr" \
				   -DCMAKE_C_COMPILER=/usr/bin/gcc \
				   -DCMAKE_CXX_STANDARD=17 \
				   -DFAISS_OPT_LEVEL=avx512 \
				   -DFAISS_ENABLE_C_API=ON \
				   -DCUDAToolkit_ROOT=/usr/local/cuda \
				   -DCMAKE_BUILD_TYPE=Release .

cmake -B build_gpu_cuvs -DFAISS_ENABLE_GPU=ON \
				   -DFAISS_ENABLE_PYTHON=OFF \
				   -DBUILD_SHARED_LIBS=ON \
				   -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
				   -DCMAKE_CUDA_FLAGS="-Xcompiler=-std=c++17 --expt-relaxed-constexpr" \
				   -DCMAKE_C_COMPILER=/usr/bin/gcc \
				   -DCMAKE_CXX_STANDARD=17 \
				   -DFAISS_OPT_LEVEL=avx512 \
				   -DFAISS_ENABLE_C_API=ON \
				   -DCUDAToolkit_ROOT=/usr/local/cuda \
				   -DFAISS_ENABLE_CUVS=ON \
				   -DCMAKE_BUILD_TYPE=Release .

g++ -o faiss_demo faiss_demo.cpp     -lfaiss     -L/usr/local/cuda/lib64     -lcudart -lcublas     -std=c++17 -O3 -fopenmp 

g++ -o demo demo.cpp     -lfaiss     -L/usr/local/cuda/lib64     -lcudart -lcublas  -lopenblas   -std=c++17 -O3 -fopenmp


cmake -B build_gpu_c \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_STANDARD=17 \
  -DFAISS_OPT_LEVEL=avx512 \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_BUILD_TYPE=Release .

gcc -o c_demo demo.c -L/usr/local/lib  -L/usr/local/cuda/lib64     -lcudart -lcublas  -lopenblas -lfaiss_c  -lm  -lpthread -O3 -fopenmp