
opencv_home=/root/projects/deps/opencv3gcc4.8/
export PATH=${opencv_home}/bin:$PATH
export LD_LIBRARY_PATH=${opencv_home}/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=${opencv_home}/include:${opencv_home}/include/opencv:${opencv_home}/include/opencv2:${CPLUS_INCLUDE_PATH}
export PATH=/opt/compiler/gcc-4.8.2/bin/:${PATH}

WITH_GPU=ON
PADDLE_DIR=/root/projects/deps/fluid_inference/
CUDA_LIB=/usr/local/cuda-10.0/lib64/
CUDNN_LIB=/usr/local/cuda-10.0/lib64/
OPENCV_DIR=/root/projects/deps/opencv341/
export LD_LIBRARY_PATH=${OPENCV_DIR}/lib:$LD_LIBRARY_PATH

rm -rf build
mkdir -p build
cd build

cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_STATIC_LIB=OFF
make clean
make -j12
