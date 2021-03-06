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

void convolute_forward(Matrix *input, Matrix *weight, Matrix *output) {
    //Aux variables
    uint on, om, wn, wm;
    //Output loop
    for(on = 0; on < output->n; on++)
    for(om = 0; om < output->m; om++) {
        //Weight matrix loop - KERNEL
        for(wn = 0; wn < weight->n; wn++)
        for(wm = 0; wm < weight->m; wm++)
            //Cross-Correlation
            MATRIX_VALUE(output, on, om) += MATRIX_VALUE(input, (on + wn), (om + wm)) * MATRIX_VALUE(weight, wn, wm);
    }
}

void convolution_forward(Feature **input, LeNet lenet) {
    Feature *output = *(input + 1);
    //Aux variables
    uint wn, wm;
    //Convolution
    for(wn = 0; wn < lenet.weight->n; wn++)
    for(wm = 0; wm < lenet.weight->m; wm++)
        convolute_forward(FEATURE_GETMATRIX(*input, wn), WEIGHT_GETMATRIX(lenet.weight, wn, wm), 
                          FEATURE_GETMATRIX(output, wm));
    //Activation function
    activation_forward(output, lenet.bias, ReLU);
}

void subsampling_forward(Feature **input) {
    Feature *output = *(input + 1);
    //Aux variables
    Matrix *inputMatrix, *outputMatrix;
    uint o, on, om, ln, lm, aux_n, aux_m;
    number max, aux;
    const uint lnLength = FEATURE_GETMATRIX(*input, 0)->n / FEATURE_GETMATRIX(output, 0)->n,
               lmLength = FEATURE_GETMATRIX(*input, 0)->m / FEATURE_GETMATRIX(output, 0)->m;
    //Ouput array loop
    for(o = 0; o < output->n; o++) {
        inputMatrix = FEATURE_GETMATRIX(*input, o);
        outputMatrix = FEATURE_GETMATRIX(output, o);
        //Output matrix loop
        for(on = 0; on < outputMatrix->n; on++)
        for(om = 0; om < outputMatrix->m; om++) {
            //Subsampling
            max = -1, aux_n = lnLength*on, aux_m = lmLength*om;
            for(ln = 0; ln < lnLength; ln++)
            for(lm = 0; lm < lmLength; lm++) {
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
    const uint wn1Length = (*input)->n, wn2Length = (weightMatrix->n)/wn1Length;
    //Dot product
    for(wn1 = 0; wn1 < wn1Length; wn1++) {
        inputMatrix = FEATURE_GETMATRIX(*input, wn1);
        wn1_aux = wn1*wn2Length;
        for(wn2 = 0; wn2 < wn2Length; wn2++)
        for(wm = 0; wm < weightMatrix->m; wm++)
            MATRIX_VALUE1(outputMatrix, wm) += MATRIX_VALUE1(inputMatrix, wn2) * MATRIX_VALUE(weightMatrix, (wn1_aux + wn2), wm);
    }
    //Activation function
    for(wm = 0; wm < lenet.bias->n; wm++)
        MATRIX_VALUE1(outputMatrix, wm) = ReLU(MATRIX_VALUE1(outputMatrix, wm) + ARRAY_VALUE(lenet.bias, wm));
}
