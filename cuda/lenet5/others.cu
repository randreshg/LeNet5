#include "lenet.h"

/* ----- OTHERS FUNCTIONS ----- */
number ReLU(number x) {
    return x*(x > 0);
}

number ReLU_GRAD(number x) {
    return x > 0;
}

void softMax(Feature *input, uint8 target, Feature *featureGradient) {
    //int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tn = threadIdx.y, tm = threadIdx.x;
    //int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = d_in[myId];
    }
    //Aux variables
    uint8 on, om;
    number den = 0, inner = 0;
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    Matrix *gradientMatrix = FEATURE_GETMATRIX(featureGradient, 0);
    //Error and softmax
    for(on = 0; on < inputMatrix->m; on++) {
        den = 0;
        for(om = 0; om < inputMatrix->m; om++)
            den += exp(MATRIX_VALUE1(inputMatrix, om) - MATRIX_VALUE1(inputMatrix, on));
        ARRAY_VALUE(gradientMatrix, on) = 1.0/den;
        inner -= ARRAY_VALUE(gradientMatrix, on) * ARRAY_VALUE(gradientMatrix, on);
    }
    inner += ARRAY_VALUE(gradientMatrix, target);
    for(om = 0; om < gradientMatrix->m; om++)
        ARRAY_VALUE(gradientMatrix, om) *= (om == target) - MATRIX_VALUE1(gradientMatrix, om) - inner;
}

number costFunction(Feature *input, uint8 target) {
    //Aux variables
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    return (-log(MATRIX_VALUE(FEATURE_GETMATRIX(input, 0), 0, target))/inputMatrix->m);
}
