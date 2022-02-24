#ifndef HELPER_DEVTRANSFER_H
#define HELPER_DEVTRANSFER_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <assert.h>
#include <helper_cuda.h>

/* transfer a 1-D vector data to device and return the device pointer to out_ptr */
template<typename T>
void toDevice(std::vector<T> data, T** out_d_ptr){
    // allocate device memory
    T* d_ptr;
    checkCudaErrors( cudaMalloc((void**)&d_ptr, data.size()*sizeof(T)) );
    // transfer
    T* h_ptr = &data[0];
    checkCudaErrors( cudaMemcpy(d_ptr, h_ptr, data.size()*sizeof(T), cudaMemcpyHostToDevice) );

    *out_d_ptr = d_ptr;
}


/* transfer a 1-D vector data from device to host
    if out_vec is not null, copy data to the memory location there
    otherwise, allocate and write the memory location to out_vec
    n_data: number of element on array d_data (assert n_data <= out_vec.size)
 */
template<typename T>
void toHost(T* d_data, int n_data, std::vector<T> *out_vec){
    if (out_vec){
        assert  (n_data <= (*out_vec).size()) ;
    } else {
        std::vector<T> tmp(n_data, 0);
        *out_vec = tmp;
    }

    checkCudaErrors( cudaMemcpy(&(*out_vec)[0], d_data, n_data*sizeof(T), cudaMemcpyDeviceToHost ) );
}


#endif // include shield