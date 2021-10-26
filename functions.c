#include "functions.h"
#include "lenet.h"
#include <stdlib.h>

/* ----- FUNCTIONS ----- */
#define ReLU(x) (x>0? x: 0)

void convolute(float **input, float **weight, float **output ){
    int ox, oy, wx, wy;
    int ox_size = GET_LENGTH(output), oy_size = GET_LENGTH(*output);
    int wx_size = GET_LENGTH(weight), wy_size = GET_LENGTH(*weight);
    //Output loop
    for(ox = 0; ox < ox_size; ox++)
        for(oy = 0; oy < oy_size; oy++)
        //Weight matrix loop
        for(wx = 0; wx < wx_size; wx++)
            for(wy = 0; wy < wy_size; wy++)
            //Cross-Correlation
            *(*(output+ox)+oy) += (*(*(input+ox+wx)+oy+wy)) * (*(*(weight+wx)+wy));
}

void convolution(float ***input, float ****weight, float *bias, float ***output){
    int wx, wy;
    int wx_size = GET_LENGTH(weight), wy_size = GET_LENGTH(*weight);
    //Convolution
    for(wx = 0; wx < wx_size; wx++)
        for(wy = 0; wy < wy_size; wy++)
            convolute(*(input+wx), *(*(weight+wx)+wy), *(output+wy));
    //Activation function + bias
    int ox_size = GET_LENGTH(output), oy_size = GET_LENGTH(*output);
    for(wx = 0; wx < ox_size; wx++)
        for(wy = 0; wy < oy_size; wy++)
            *(**(output+wx)+wy) = ReLU(*(**(output+wx)+wy) + *(bias+wx));
}

void subsampling(float ***input, float ***output){
    int o, ox, oy, lx, ly, max, aux_x, aux_y, aux;
    int o_size = GET_LENGTH(output), ox_size = GET_LENGTH(*output), oy_size = GET_LENGTH(**output);
    int lx_size = GETLENGTH(*input)/ox_size, ly_size = GETLENGTH(**input)/oy_size;
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

void dotproduct(float ***input, float **weight, float *bias, float *output){
    int wx, wy;
    int wx_size = GET_LENGTH(weight), wy_size = GET_LENGTH(*weight);
    //Convolution
    for(wx = 0; wx < wx_size; wx++)
        for(wy = 0; wy < wy_size; wy++)
            *(output + wy) += ((float *)input)[wx] * (*(*(weight + wx) + wy));
    //Activation function + bias
    int ox_size = GET_LENGTH(bias);
    for(wx = 0; wx < ox_size; wx++)
        *(output + wx) = ReLU(*(output + wx) + *(bias + wx));
}
