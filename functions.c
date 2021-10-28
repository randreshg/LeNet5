#include "functions.h"
#include "lenet.h"
#include <stdlib.h>

/* ----- FUNCTIONS ----- */
#define ReLU(x) (x>0? x: 0)

void convolute(double *input, double *weight, double **output ){
    //Aux variables
    const int ox_size = GET_LENGTH(output), oy_size = GET_LENGTH(*output);
    //const int wn_size = GET_LENGTH(weight), wm_size = GET_LENGTH(*weight);
    int ox, oy, wn, wm;
    //Output loop
    for(ox = 0; ox < ox_size; ox++)
    for(oy = 0; oy < oy_size; oy++)
        //Weight matrix loop
        for(wn = 0; wn < LENGTH_KERNEL; wn++)
        for(wm = 0; wm < LENGTH_KERNEL; wm++)
            //Cross-Correlation
            *(*(output+ox)+oy) += (*(*(input+ox+wn)+oy+wm)) * (*(*(weight+wn)+wm));
}

void convolution(Feature *input, LeNet lenet){
    Feature *output = input + 1;
    MALLOC_FEATURE(output);
    //Aux variables
    uint8 wn, wm;
    //Convolution
    for(wn = 0; wn < lenet.weight->n; wn++)
    for(wm = 0; wm < lenet.weight->m; wm++)
        convolute(input+wn, *(*(weight+wn)+wm), output+wm);
    //Activation function + bias
    // int ox_size = GET_LENGTH(output), oy_size = GET_LENGTH(*output);
    // for(wn = 0; wn < ox_size; wn++)
    // for(wm = 0; wm < oy_size; wm++)
    //     *(**(output+wn)+wm) = ReLU(*(**(output+wn)+wm) + *(bias+wn));
    //Free memory
}

void subsampling(double ***input, double ***output){
    int o, ox, oy, lx, ly, max, aux_x, aux_y, aux;
    const int o_size = GET_LENGTH(output), ox_size = GET_LENGTH(*output), oy_size = GET_LENGTH(**output);
    const int lx_size = GET_LENGTH(*input)/ox_size, ly_size = GET_LENGTH(**input)/oy_size;
    for(o = 0; o < o_size; o++)
        for(ox = 0; ox < ox_size; ox++)
            for(oy = 0; oy < oy_size; oy++)
            {
                max = 0, aux_x = lx_size*ox, aux_y = ly_size*oy;
                for(lx = 0; lx < lx_size; lx++)
                    for(ly = 0; ly < ly_size; ly++){
                        aux = *(*(*(input + o) + aux_x + lx) + aux_y + ly);
                        max = aux > max ? aux:max;
                    }
                *(*(*(output + o) + ox) + oy) = max;
            }
}

void dotproduct(double ***input, double **weight, double *bias, double *output){
    int wn, wm;
    const int wn_size = GET_LENGTH(weight), wm_size = GET_LENGTH(*weight);
    //Convolution
    for(wn = 0; wn < wn_size; wn++)
        for(wm = 0; wm < wm_size; wm++)
            *(output + wm) += ((double *)input)[wn] * (*(*(weight + wn) + wm));
    //Activation function + bias
    const int ox_size = GET_LENGTH(bias);
    for(wn = 0; wn < ox_size; wn++)
        *(output + wn) = ReLU(*(output + wn) + *(bias + wn));
}
