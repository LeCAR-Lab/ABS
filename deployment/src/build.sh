rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
make
