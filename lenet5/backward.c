#include "lenet.h"

/* ----- BACKWARD FUNCTIONS ----- */
void activation_backward(Feature *output, number (*action)(number)){
    uint on, om, matrixSize;
    Matrix *outputMatrix;
    for(on = 0; on < output->n; on++){
        outputMatrix = FEATURE_GETMATRIX(output, on);
        matrixSize = MATRIX_SIZE(outputMatrix);
        for(om = 0; om < matrixSize; om++)
            MATRIX_VALUE1(outputMatrix, om) *= action(MATRIX_VALUE1(outputMatrix, om));
    }
}

void convolute_backward(Matrix *input, Matrix *weight, Matrix *output ){
    //Aux variables
    uint in, im, wn, wm;
    //Input loop
    for(in = 0; in < input->n; in++)
    for(im = 0; im < input->m; im++){
        //Weight matrix loop
        for(wn = 0; wn < weight->n; wn++)
        for(wm = 0; wm < weight->m; wm++)
            //Cross-Correlation
            MATRIX_VALUE(output, (in+wn), (im+wm)) += MATRIX_VALUE(input, in, im) * MATRIX_VALUE(weight, wn, wm);
    }
}

void convolution_backward(Feature *input, LeNet lenet, Feature **inputGradient, LeNet *lenetGradient){
    Feature *outputGradient = *(inputGradient - 1);
    //Aux variables
    uint wn, wm, matrixSize;
    //Calculate output gradient
    for(wn = 0; wn < lenet.weight->n; wn++)
    for(wm = 0; wm < lenet.weight->m; wm++)
        convolute_backward(FEATURE_GETMATRIX(*inputGradient, wm), WEIGHT_GETMATRIX(lenet.weight, wn, wm), 
                           FEATURE_GETMATRIX(outputGradient, wn));
    //Activation function
    activation_backward(outputGradient, ReLU_GRAD);
    //Update bias
    Matrix *auxMatrix;
    for(wn = 0; wn < (*inputGradient)->n; wn++){
        auxMatrix = FEATURE_GETMATRIX(*inputGradient, wn);
        matrixSize = MATRIX_SIZE(auxMatrix);
        for(wm = 0; wm < matrixSize; wm++)
            ARRAY_VALUE(lenetGradient->bias, wn) += MATRIX_VALUE1(auxMatrix, wm);
    }
    //Update weights
    for(wn = 0; wn < lenet.weight->n; wn++)
    for(wm = 0; wm < lenet.weight->m; wm++)
        convolute_forward(FEATURE_GETMATRIX(input, wn), FEATURE_GETMATRIX(*inputGradient, wm),
                          WEIGHT_GETMATRIX(lenetGradient->weight, wn, wm));
}

void subsampling_backward(Feature *input, Feature **inputGradient){
    Feature *outputGradient = *(inputGradient - 1);
    //Aux variables
    Matrix *inputMatrix, *inputGradientMatrix, *outputGradientMatrix;
    uint i, in, im, ln, lm, maxLn=0, maxLm=0, aux_n, aux_m;
    number max, aux;
    const uint ln_length = FEATURE_GETMATRIX(outputGradient, 0)->n / FEATURE_GETMATRIX(*inputGradient, 0)->n,
               lm_length = FEATURE_GETMATRIX(outputGradient, 0)->m / FEATURE_GETMATRIX(*inputGradient, 0)->m;
    //Input array loop
    for(i = 0; i < (*inputGradient)->n; i++){
        inputMatrix = FEATURE_GETMATRIX(input, i);
        inputGradientMatrix = FEATURE_GETMATRIX(*inputGradient, i);
        outputGradientMatrix = FEATURE_GETMATRIX(outputGradient, i);
        //Input matrix loop
        for(in = 0; in < inputGradientMatrix->n; in++)
        for(im = 0; im < inputGradientMatrix->m; im++){
            //Subsampling
            max = -1.0, aux_n = ln_length*in, aux_m = lm_length*im;
            for(ln = 0; ln < ln_length; ln++){
                for(lm = 0; lm < lm_length; lm++){
                    aux = MATRIX_VALUE(inputMatrix, (aux_n + ln), (aux_m + lm));
                    if(aux > max)
                        max = aux, maxLn = (aux_n + ln), maxLm = (aux_m + lm);
                }
            }
            MATRIX_VALUE(outputGradientMatrix, maxLn, maxLm) = MATRIX_VALUE(inputGradientMatrix, in, im);
        }
    }
}

void dotproduct_backward(Feature *input, LeNet lenet, Feature **inputGradient, LeNet *lenetGradient){
    Feature *outputGradient = *(inputGradient - 1);
    //Aux variables
    uint wn1, wn2, wm, wn1_aux;
    Matrix *auxMatrix;
    Matrix *weightMatrix = WEIGHT_GETMATRIX1(lenet.weight, 0);
    Matrix *weightGradientMatrix = WEIGHT_GETMATRIX1(lenetGradient->weight, 0);
    Matrix *inputGradientMatrix = FEATURE_GETMATRIX(*inputGradient, 0);
    //Constants
    const uint wn1_length = outputGradient->n, wn2_length = (weightMatrix->n)/wn1_length;
    //Dot product + activation function
    for(wn1 = 0; wn1 < wn1_length; wn1++){
        auxMatrix = FEATURE_GETMATRIX(outputGradient, wn1);
        wn1_aux = wn1 * wn2_length;
        for(wn2 = 0; wn2 < wn2_length; wn2++){
            //Dot product
            for(wm = 0; wm < weightMatrix->m; wm++)
                MATRIX_VALUE1(auxMatrix, wn2) += MATRIX_VALUE1(inputGradientMatrix, wm) * MATRIX_VALUE(weightMatrix, wn1_aux + wn2, wm);
        }
    }
    //Activation function
    activation_backward(outputGradient, ReLU_GRAD);
    //Update bias
    for(wm = 0; wm < (lenetGradient->bias)->n; wm++)
        ARRAY_VALUE(lenetGradient->bias, wm) += MATRIX_VALUE1(inputGradientMatrix, wm);
    //Update weights
    auxMatrix = FEATURE_GETMATRIX(input, 0);
    for(wn1 = 0; wn1 < wn1_length; wn1++){
        wn1_aux = wn1 * wn2_length;
        for(wn2 = 0; wn2 < wn2_length; wn2++){
            //Dot product
            for(wm = 0; wm < weightMatrix->m; wm++)
                MATRIX_VALUE(weightGradientMatrix, wn1_aux + wn2, wm) += MATRIX_VALUE1(auxMatrix, wn2) * MATRIX_VALUE1(inputGradientMatrix, wm);
        }
    }
}