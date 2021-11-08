#include "lenet.h"

/* ----- FORWARD FUNCTIONS ----- */
void activation_forward(Feature *output, Array *bias){
    uint wn, wm, matrixSize;
    Matrix *outputMatrix;
    for(wn = 0; wn < output->n; wn++){
        outputMatrix = FEATURE_GETMATRIX(output, wn);
        matrixSize = MATRIX_SIZE(outputMatrix);
        for(wm = 0; wm < matrixSize; wm++)
            MATRIX_VALUE1(outputMatrix, wm) = ReLU(MATRIX_VALUE1(outputMatrix, wm) + ARRAY_VALUE(bias, wn));
    }
}

void convolute_forward(Matrix *input, Matrix *weight, Matrix *output ){
    //Aux variables
    uint on, om, wn, wm;
    //Output loop
    for(on = 0; on < output->n; on++)
    for(om = 0; om < output->m; om++){
        MATRIX_VALUE(output, on, om) = 0;
        //Weight matrix loop - KERNEL
        for(wn = 0; wn < weight->n; wn++)
        for(wm = 0; wm < weight->m; wm++)
            //Cross-Correlation
            MATRIX_VALUE(output, on, om) += MATRIX_VALUE(input, (on+wn), (om+wm)) * MATRIX_VALUE(weight, wn, wm);
    }
}

void convolution_forward(Feature *input, LeNet lenet){
    Feature *output = input + 1;
    //Aux variables
    uint wn, wm;
    //Convolution
    for(wn = 0; wn < lenet.weight->n; wn++)
    for(wm = 0; wm < lenet.weight->m; wm++)
        convolute_forward(FEATURE_GETMATRIX(input, wn), WEIGHT_GETMATRIX(lenet.weight, wn, wm), 
                          FEATURE_GETMATRIX(output, wm));
    //Activation function
    activation_forward(output, lenet.bias);
}

void subsampling_forward(Feature *input){
    Feature *output = input + 1;
    //Aux variables
    Matrix *mo;
    unsigned int o, on, om, ln, lm, max, aux_n, aux_m, aux;
    const uint ln_length = (input->matrix->n)/(output->matrix->n), lm_length = (input->matrix->m)/(output->matrix->m);
    //Ouput array loop
    for(o = 0; o < output->n; o++){
        mo = FEATURE_GETMATRIX(output, o);
        //Output matrix loop
        for(on = 0; on < mo->n; on++)
        for(om = 0; om < mo->m; om++){
            //Subsampling
            max = 0, aux_n = ln_length*on, aux_m = lm_length*om;
            for(ln = 0; ln < ln_length; ln++)
                for(lm = 0; lm < lm_length; lm++){
                    aux = MATRIX_VALUE(FEATURE_GETMATRIX(input, o), (aux_n + ln), (aux_m + lm));
                    max = aux > max ? aux:max;
                }
            MATRIX_VALUE(mo, on, om) = max;
        }
    }
}

void dotproduct_forward(Feature *input, LeNet lenet){
    Feature *output = input + 1;
    //Aux variables
    uint wn1, wn2, wm, wn1_aux;
    Matrix *inputMatrix;
    Matrix *weightMatrix = WEIGHT_GETMATRIX(lenet.weight, 0, 0);
    Matrix *outputMatrix = FEATURE_GETMATRIX(output, 0);
    const uint wn1_length = input->n, wn2_length = (weightMatrix->n)/wn1_length;
    //Dot product
    for(wn1 = 0; wn1 < wn1_length; wn1++){
        inputMatrix = FEATURE_GETMATRIX(input, wn1);
        wn1_aux = wn1*wn2_length;
        for(wn2 = 0; wn2 < wn2_length; wn2++)
        for(wm = 0; wm < weightMatrix->m; wm++)
            MATRIX_VALUE1(outputMatrix, wm) += MATRIX_VALUE1(inputMatrix, wn2) * MATRIX_VALUE(weightMatrix, (wn1_aux+wn2), wm);
    }
    //Activation function
    activation_forward(output, lenet.bias);
}
