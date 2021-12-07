#include "lenet.h"

/* ----- FORWARD FUNCTIONS ----- */
void activation_forward(Feature *output, Array *bias, number (*action)(number)) {
    uint on, om, matrixSize;
    Matrix *outputMatrix;
    for(on = 0; on < output->n; on++) {
        outputMatrix = FEATURE_GETMATRIX(output, on);
        matrixSize = MATRIX_SIZE(outputMatrix);
        for(om = 0; om < matrixSize; om++)
            MATRIX_VALUE1(outputMatrix, om) = action(MATRIX_VALUE1(outputMatrix, om) + ARRAY_VALUE(bias, on));
    }
}

__global__ void convolute_forward(Matrix *input, Matrix *weight, Matrix *output) {
    uint tn = threadIdx.y, tm = threadIdx.x;
    uint tid = tn*blockDim.x + tm;
    //Shared memory
    __shared__ number s_input[blockDim.y][blockDim.x];
    __shared__ number s_weight[weight->n][weight->m];
    //Load shared mem from global mem
    s_input[tn][tm] = MATRIX_VALUE(input, tn, tm);
    if(tid < (weight->n*weight->m))
        s_weight[tn][tm] = MATRIX_VALUE(weight, tn, tm);
    __syncthreads();
    //Thread id inside output matrix
    if(tid < (output->n*output->m)) {
        //Aux variables
        uint wn, wm;
        number result = 0;
        //Weight matrix loop - KERNEL
        for(wn = 0; wn < weight->n; wn++)
        for(wm = 0; wm < weight->m; wm++)
            result += s_input[(tn + wn), (tm + wm)]) * s_weight[wn, wm];
        //Write back result
        atomicAdd(MATRIX_POINTER(output, tn, tm), result);
    }
}

__global__ void convolution_forward(Feature **input, LeNet lenet) {
    uint bn = blockIdx.y,  bm = blockIdx.x;
    uint tn = threadIdx.y, tm = threadIdx.x;
    uint tid = tn*blockDim.x + tm;
    //Aux variables
    Matrix *input  = FEATURE_GETMATRIX(*input, bn),
           *output = FEATURE_GETMATRIX(*(input + 1), bm),
           *weight = WEIGHT_GETMATRIX(lenet.weight, bn, bm);
    // ---- Convolution ---- //
    //Shared memory
    __shared__ number s_input[blockDim.y][blockDim.x];
    __shared__ number s_weight[weight->n][weight->m];
    //Load shared mem from global mem
    s_input[tn][tm] = MATRIX_VALUE(input, tn, tm);
    if(tid < (weight->n*weight->m))
        s_weight[tn][tm] = MATRIX_VALUE(weight, tn, tm);
    __syncthreads();
    //Thread id inside output matrix
    if(tid < (output->n*output->m)) {
        //Aux variables
        uint wn, wm;
        number result = 0;
        //Weight matrix loop - KERNEL
        for(wn = 0; wn < weight->n; wn++)
        for(wm = 0; wm < weight->m; wm++)
            result += s_input[(tn + wn), (tm + wm)]) * s_weight[wn, wm];
        //Write back result
        atomicAdd(MATRIX_POINTER(output, tn, tm), result);
    }

    //Activation function
    //activation_forward(output, lenet.bias, ReLU);
}

void subsampling_forward(Feature **input) {
    Feature *output = *(input + 1);
    //Aux variables
    Matrix *inputMatrix, *outputMatrix;
    uint o, on, om, ln, lm, aux_n, aux_m;
    number max, aux;
    const uint ln_length = FEATURE_GETMATRIX(*input, 0)->n / FEATURE_GETMATRIX(output, 0)->n,
               lm_length = FEATURE_GETMATRIX(*input, 0)->m / FEATURE_GETMATRIX(output, 0)->m;
    //Ouput array loop
    for(o = 0; o < output->n; o++) {
        inputMatrix = FEATURE_GETMATRIX(*input, o);
        outputMatrix = FEATURE_GETMATRIX(output, o);
        //Output matrix loop
        for(on = 0; on < outputMatrix->n; on++)
        for(om = 0; om < outputMatrix->m; om++) {
            //Subsampling
            max = -1, aux_n = ln_length*on, aux_m = lm_length*om;
            for(ln = 0; ln < ln_length; ln++)
            for(lm = 0; lm < lm_length; lm++) {
                aux = MATRIX_VALUE(inputMatrix, (aux_n + ln), (aux_m + lm));
                max = (aux > max) ? aux:max;
            }
            MATRIX_VALUE(outputMatrix, on, om) = max;
        }
    }
}

void dotproduct_forward(Feature **input, LeNet lenet) {
    Feature *output = *(input + 1);
    //Aux variables
    uint wn1, wn2, wm, wn1_aux;
    Matrix *inputMatrix, 
           *weightMatrix = WEIGHT_GETMATRIX1(lenet.weight, 0),
           *outputMatrix = FEATURE_GETMATRIX(output, 0);
    const uint wn1_length = (*input)->n, wn2_length = (weightMatrix->n)/wn1_length;
    //Dot product
    for(wn1 = 0; wn1 < wn1_length; wn1++) {
        inputMatrix = FEATURE_GETMATRIX(*input, wn1);
        wn1_aux = wn1*wn2_length;
        for(wn2 = 0; wn2 < wn2_length; wn2++)
        for(wm = 0; wm < weightMatrix->m; wm++)
            MATRIX_VALUE1(outputMatrix, wm) += MATRIX_VALUE1(inputMatrix, wn2) * MATRIX_VALUE(weightMatrix, (wn1_aux + wn2), wm);
    }
    //Activation function
    for(wm = 0; wm < lenet.bias->n; wm++)
        MATRIX_VALUE1(outputMatrix, wm) = ReLU(MATRIX_VALUE1(outputMatrix, wm) + ARRAY_VALUE(lenet.bias, wm));
}
