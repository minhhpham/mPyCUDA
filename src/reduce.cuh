#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <helper_devTransfer.h>
#include <cub/cub.cuh>


double sumReduce(std::vector<double> data){
    // transfer data to device
    double *d_data;
    toDevice(data, &d_data);
    // allocate output
    double result, *d_result;
    checkCudaErrors( cudaMalloc(&d_result, sizeof(double)) );

    // sum reduce on device
        // determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_data, d_result, data.size());
    checkCudaKernelErrors();
        // Allocate temporary storage
    checkCudaErrors( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
        // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_data, d_result, data.size());
    checkCudaKernelErrors();

    // transfer result to host
    checkCudaErrors( cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost) );

    cudaFree(d_data); cudaFree(d_temp_storage); cudaFree(d_result);
    return result;
}