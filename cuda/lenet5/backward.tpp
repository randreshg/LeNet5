#include "lenet.h"

/* ----- BACKWARD FUNCTIONS ----- */
template<size_t IN, size_t IN1, size_t IM1, size_t ON, size_t ON1, size_t OM1>
__global__ void activation_backward(number (&input)[IN][IN1][IM1], number (&output)[ON][ON1][OM1]) {
    uint bn = blockIdx.x, tn = threadIdx.x;
    ((number *)output[bn])[tn] *= (((number *)input[bn])[tn] > 0);
}

template<size_t IN, size_t IN1, size_t IM1, size_t IGN, size_t IGN1, size_t IGM1,
         size_t WN, size_t WM,  size_t WN1, size_t WM1, size_t WGN, size_t WGM, size_t WGN1, size_t WGM1, size_t BG1,
         size_t OGN, size_t OGN1, size_t OGM1>
__global__ void convolution_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN][IGN1][IGM1],
                                     number (&weight)[WN][WM][WN1][WM1], number (&weightGradient)[WGN][WGM][WGN1][WGM1], number (&biasGradient)[BG1],
                                     number (&outputGradient)[OGN][OGN1][OGM1]) {
    uint bn = blockIdx.x,  bm = blockIdx.y;
    uint tn = threadIdx.x, tm = threadIdx.y;
    //Aux variables
    uint wn, wm;
    //Shared memory
    __shared__ number s_input[IN1][IM1], s_inputG[IGN1][IGM1];
    __shared__ number s_weight[WN1][WM1];
    //Load shared mem from global mem
    s_input[tn][tm] = input[bn][tn][tm];
    if(tn < IGN1 && tm < IGM1)
        s_inputG[tn][tm] = inputGradient[bm][tn][tm];
    if(tn < WN1 && tm < WM1)
        s_weight[tn][tm] = weight[bn][bm][tn][tm];
    __syncthreads();
    
    if(tn < IGN1 && tm < IGM1) {
        //Convolution backwards
        for(wn = 0; wn < WN1; wn++)
        for(wm = 0; wm < WM1; wm++)
            atomicAdd(&(outputGradient[bn][tn + wn][tm + wm]), (s_inputG[tn][tm] * s_weight[wn][wm]));
        //Update bias
        if(bn == 0)
            atomicAdd(&(biasGradient[bm]), s_inputG[tn][tm]);
    }
    //Update weights
    if(tn < WGN1 && tm < WGM1) {
        number result = 0;
        for(wn = 0; wn < IGN1; wn++)
        for(wm = 0; wm < IGM1; wm++)
            result += s_input[tn + wn][tm + wm] * s_inputG[wn][wm];
        atomicAdd(&(weightGradient[bn][bm][tn][tm]), result);
    }
}

template<size_t IN, size_t IN1, size_t IM1, size_t IGN, size_t IGN1, size_t IGM1, size_t OGN, size_t OGN1, size_t OGM1>
__global__ void subsampling_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN][IGN1][IGM1], number (&outputGradient)[OGN][OGN1][OGM1]) {
    uint bn = blockIdx.x, tn = threadIdx.x, tm = threadIdx.y;
    //Shared memory
    __shared__ number s_input[IN1][IM1];
    //Load shared mem from global mem
    s_input[tn][tm] = input[bn][tn][tm];
    __syncthreads();
    if(tn < IGN1 && tm < IGM1) {
        //Aux variables
        const uint lnLength = OGN1/IGN1, lmLength = OGM1/IGM1;
        uint ln, lm, maxLn, maxLm, aux_n, aux_m;
        number max, aux;
        max = -1.0, aux_n = lnLength*tn, aux_m = lmLength*tm;
        //Subsampling
        for(ln = 0; ln < lnLength; ln++) 
        for(lm = 0; lm < lmLength; lm++) {
            aux = s_input[aux_n + ln][aux_m + lm];
            if(aux > max)
                max = aux, maxLn = (aux_n + ln), maxLm = (aux_m + lm);
        }
        outputGradient[bn][maxLn][maxLm] = inputGradient[bn][tn][tm];
    }
}

template<size_t IN, size_t IN1, size_t IM1, size_t IGN,
         size_t WN, size_t WM, size_t WGN, size_t WGM, size_t BG1,
         size_t OGN, size_t OGN1, size_t OGM1>
__global__ void dotproduct_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN],
                                    number (&weight)[WN][WM], number (&weightGradient)[WGN][WGM], number (&biasGradient)[BG1],
                                    number (&outputGradient)[OGN][OGN1][OGM1]) {
    uint tn = threadIdx.x, tm = threadIdx.y, tn1 = tn + blockDim.x;
    //Dot product
    atomicAdd(&(((number *)outputGradient)[tn]),  (inputGradient[tm] * weight[tn][tm]));
    atomicAdd(&(((number *)outputGradient)[tn1]), (inputGradient[tm] * weight[tn1][tm]));
    __syncthreads();
    //Activation function
    if(tm == 0) {
        ((number *)outputGradient)[tn] *= (((number *)input)[tn] > 0);
        ((number *)outputGradient)[tn1] *= (((number *)input)[tn1] > 0);
    }
    //Update bias
    if(tn == 0)
        biasGradient[tm] += inputGradient[tm];
    //Update weights
    atomicAdd(&(weightGradient[tn][tm]),  (inputGradient[tm] * ((number *)input)[tn]));
    atomicAdd(&(weightGradient[tn1][tm]), (inputGradient[tm] * ((number *)input)[tn1]));
}