set -e  # Exit on any error

cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=native" \
      -DCMAKE_C_FLAGS="-march=native" \
      -DSZ3_DEBUG_TIMINGS=1 \
      -DCMAKE_INSTALL_PREFIX=.. ..

# Build
make clean
make -j$(nproc)  # Use all cores
make install
