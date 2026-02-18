mkdir build
cd build
cmake -DNORM_COSINE=ON -DCMAKE_BUILD_TYPE=Debug ..
make
cd ..