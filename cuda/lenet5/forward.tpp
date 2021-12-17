#include "lenet.h"

/* ----- FORWARD FUNCTIONS ----- */
template<size_t ON, size_t ON1, size_t OM1, size_t BN>
__global__ void activation_forward(number (&output)[ON][ON1][OM1], number (&bias)[BN]) {
    uint bn = blockIdx.x, tn = threadIdx.x;
    //ReLu
    number value = ((number *)output[bn])[tn] + bias[bn];
    ((number *)output[bn])[tn] = value*(value > 0);
}

template<size_t IN, size_t IN1, size_t IM1, size_t WN, size_t WM, size_t WN1, size_t WM1, size_t ON, size_t ON1, size_t OM1>
__global__ void convolution_forward(number (&input)[IN][IN1][IM1], number (&weight)[WN][WM][WN1][WM1], number (&output)[ON][ON1][OM1]) {
    uint bn = blockIdx.x,  bm = blockIdx.y;
    uint tn = threadIdx.x, tm = threadIdx.y;
    //uint tid = tn*blockDim.x + tm;
    //Shared memory
    __shared__ number s_input[IN1][IM1];
    __shared__ number s_weight[WN1][WM1];
    //Load shared mem from global mem
    s_input[tn][tm] = input[bn][tn][tm];
    if(tn < WN1 && tm < WM1)
        s_weight[tn][tm] = weight[bn][bm][tn][tm];
    __syncthreads();
    //Thread inside output matrix
    if(tn < ON1 && tm < OM1) {
        //Aux variables
        uint wn, wm;
        number result = 0;
        //Weight matrix loop - KERNEL
        for(wn = 0; wn < WN1; wn++)
        for(wm = 0; wm < WM1; wm++)
            result += s_input[tn + wn][tm + wm] * s_weight[wn][wm];
        //Write back result
        atomicAdd(&(output[bm][tn][tm]), result);
    }
}

template<size_t IN, size_t IN1, size_t IM1, size_t ON, size_t ON1, size_t OM1>
__global__ void subsampling_forward(number (&input)[IN][IN1][IM1], number (&output)[ON][ON1][OM1]) {
    uint bn = blockIdx.x, tn = threadIdx.x, tm = threadIdx.y;
    //Shared memory
    __shared__ number s_input[IN1][IM1];
    //Load shared mem from global mem
    s_input[tn][tm] = input[bn][tn][tm];
    __syncthreads();
    //Thread id inside output matrix
    if(tn < ON1 && tm < OM1) {
        //Aux variables
        const uint lnLength = IN1/ON1, lmLength = IM1/OM1;
        uint ln, lm, aux_n, aux_m;
        number max, aux;
        max = -1, aux_n = lnLength*tn, aux_m = lmLength*tm;
        //Subsampling
        for(ln = 0; ln < lnLength; ln++)
        for(lm = 0; lm < lmLength; lm++) {
            aux = s_input[aux_n + ln][aux_m + lm];
            max = (aux > max) ? aux:max;
        }
        output[bn][tn][tm] = max;
    }
}

template<size_t IN, size_t IN1, size_t IM1, size_t WN, size_t WM, size_t BN, size_t ON>
__global__ void dotproduct_forward(number (&input)[IN][IN1][IM1], number (&weight)[WN][WM], number (&bias)[BN], number (&output)[ON]) {
    uint tn = threadIdx.x, tm = threadIdx.y, tn1 = tn + blockDim.x;
    //Shared memory
    __shared__ number s_input[WN];
    //Load shared mem from global mem
    if(tm == 0) {
        s_input[tn] = ((number *)input)[tn];
        s_input[tn1] = ((number *)input)[tn1];
    }
    __syncthreads();
    //Dot product
    atomicAdd(&(output[tm]), (s_input[tn] * weight[tn][tm]) + (s_input[tn1] * weight[tn1][tm]));
    __syncthreads();
    //Activation function - ReLu
    if(tn == 0) {
        number value = output[tm] + bias[tm];
        output[tm] = value*(value > 0);
    }
}
