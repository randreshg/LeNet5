#include "lenet.h"

/* ----- OTHERS FUNCTIONS ----- */
__global__ void loadInput(uint8 *input, Features *output) {
    uint bn = blockIdx.x,  bm = blockIdx.y;
    uint tn = threadIdx.x, tm = threadIdx.y;
    uint bid = bn*IMG_SIZE, tid = tn*blockDim.x + tm;
    //Shared memory
    __shared__ number s_input[IMG_SIZE];
    __shared__ number s_mean[IMG_SIZE];
    __shared__ number s_std[IMG_SIZE];
     //Load shared mem from global mem
     s_input[tid] = (number) input[bid + tid];
     s_mean[tid] = s_input[tid];
     s_std[tid] = s_input[tid]*s_input[tid];
     __syncthreads();
     //Do reduction in shared mem
    uint s = IMG_SIZE/2;
    bool rounded = false;
    while (s > 0) {
        if (tid < s) {
            if(tid == s-1 && rounded) {
                s_mean[tid] += s_mean[tid + s] + s_mean[tid + s + 1];
                s_std[tid] += s_std[tid + s] + s_std[tid + s + 1];
            }
            else {
                s_mean[tid] += s_mean[tid + s];
                s_std[tid] += s_std[tid + s];
            }
        }
        __syncthreads();
        rounded = !(s%2)?false:true;
        s = s/2;
    }
    //Thread 0 computes mean and std
    if (tid == 0) {
        s_mean[0] = s_mean[0]/IMG_SIZE;
        s_std[0] = sqrt(s_std[0]/IMG_SIZE - s_mean[0]*s_mean[0]);
    }
    __syncthreads();
    //Write result back to global mem
    output[bn].input[0][tn + 2][tm + 2] = (s_input[tid] - s_mean[0]) / s_std[0];
}

__global__ void softMax(number input[OUTPUT], uint8 target, number outputGradient[OUTPUT]) {
    //Aux variables
    uint8 on, om;
    number den = 0, inner = 0;
    //Error and softmax
    for(on = 0; on < OUTPUT; on++) {
        den = 0;
        for(om = 0; om < OUTPUT; om++)
            den += exp(input[om] - input[on]);
        outputGradient[on] = 1.0/den;
        inner -= outputGradient[on] * outputGradient[on];
    }
    inner += outputGradient[target];
    for(on = 0; on < OUTPUT; on++)
        outputGradient[on] *= (on == target) - outputGradient[on] - inner;
}