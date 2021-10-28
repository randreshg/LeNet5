#include "functions.h"
#include "lenet.h"
#include <stdlib.h>

/* ----- FUNCTIONS ----- */
#define ReLU(x) (x>0? x: 0)

void convolute(Matrix *input, Matrix *weight, Array *bias , Matrix *output ){
    //Aux variables
    uint on, om, wn, wm;
    //Output loop
    for(on = 0; on < output->n; on++)
    for(om = 0; om < output->m; om++){
        MATRIX_VALUE(output, on, om) = 0;
        //Weight matrix loop
        for(wn = 0; wn < weight->n; wn++)
        for(wm = 0; wm < weight->m; wm++)
            //Cross-Correlation
            MATRIX_VALUE(output, on, om) += MATRIX_VALUE(input, (on+wn), (om+wm)) * MATRIX_VALUE(weight, wn, wm);
        //Activation function + bias
        MATRIX_VALUE(output, on, om) = ReLU(MATRIX_VALUE(output, on, om) + ARRAY_VALUE(bias, om));
    }
}

void convolution(Feature *input, LeNet lenet){
    //Output malloc
    Feature *output = input + 1;
    FEATURE_MALLOCMATRIX(output);
    //Aux variables
    uint wn, wm;
    //Convolution
    for(wn = 0; wn < lenet.weight->n; wn++)
    for(wm = 0; wm < lenet.weight->m; wm++)
        convolute(FEATURE_GETMATRIX(input, wn), WEIGHT_GETMATRIX(lenet.weight, wn, wm), 
                  lenet.bias, FEATURE_GETMATRIX(output, wm));
    //Free memory
    FEATURE_FREEMATRIX(input);

}

void subsampling(Feature *input){
    //Output malloc
    Feature *output = input + 1;
    FEATURE_MALLOCMATRIX(output);
    //Aux variables
    Matrix *mo;
    unsigned int o, on, om, ln, lm, max, aux_n, aux_m, aux;
    const uint ln_size = (input->matrix->n)/(output->matrix->n), lm_size = (input->matrix->m)/(output->matrix->m);
    //Ouput array loop
    for(o = 0; o < output->n; o++){
        mo = FEATURE_GETMATRIX(output, o);
        //Output matrix loop
        for(on = 0; on < mo->n; on++)
        for(om = 0; om < mo->m; om++){
            //Subsampling
            max = 0, aux_n = ln_size*on, aux_m = lm_size*om;
            for(ln = 0; ln < ln_size; ln++)
                for(lm = 0; lm < lm_size; lm++){
                    aux = MATRIX_VALUE(FEATURE_GETMATRIX(input, o), (aux_n + ln), (aux_m + lm));
                    max = aux > max ? aux:max;
                }
            MATRIX_VALUE(mo, on, om) = max;
        }
    }
    //Free memory
    FEATURE_FREEMATRIX(input);
}


void dotproduct(Feature *input, LeNet lenet){
    input->
    int wn, wm;
    const int wn_size = GET_LENGTH(weight), wm_size = GET_LENGTH(*weight);
    //Convolution
    for(wn = 0; wn < wn_size; wn++)
        for(wm = 0; wm < wm_size; wm++)
            *(output + wm) += ((double *)input)[wn] * (*(*(weight + wn) + wm));
    //Activation function + bias
    const int on_size = GET_LENGTH(bias);
    for(wn = 0; wn < on_size; wn++)
        *(output + wn) = ReLU(*(output + wn) + *(bias + wn));
}
