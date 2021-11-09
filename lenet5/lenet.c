#include "lenet.h"

LeNet *LENET(uint n, uint m, uint wm_n, uint wm_m){
    LeNet *le = (LeNet *)malloc(sizeof(LeNet));
    le->weight = WEIGHT(n, m, wm_n, wm_m);
    le->bias = ARRAY(m);
    initialValues(le);
    return le;
}

void model(){
    
}

void initialValues(LeNet *lenet){
    uint n, m, i, matrixSize;
    Matrix *matrix;
    
    for(n=0; n<lenet->weight->n; n++){
        for(m=0; m<lenet->weight->m; m++){
            matrix = *WEIGHT_GETMATRIX(lenet->weight, n, m);
            matrixSize = MATRIX_SIZE(matrix);
            for(i=0; i<matrixSize; i++){
                MATRIX_VALUE1(matrix, i) = f32Rand(10);
            }
        }
    }
    printf("OK\n");
}

void initial(LeNet **lenet){
    srand((unsigned int)time(NULL));
    lenet[0] = LENET(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[1] = LENET(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[2] = LENET(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL);
    //lenet[3] = LENET(1, 1, LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, OUTPUT);
}

void forwardPropagation(LeNet *lenet, Feature *features){
    // convolution_forward(features, *lenet);
    // subsampling_forward(features+1);
    // convolution_forward(features+2, *(lenet+1));
    // subsampling_forward(features+3);
    // convolution_forward(features+4, *(lenet+2));
    // dotproduct_forward(features+5, *(lenet+3));
}

void backwardPropagation(LeNet *lenet, Feature *features){
    // convolution_backward(features, *lenet);
    // subsampling_backward(features+1);
    // convolution_backward(features+2, *lenet);
    // subsampling_backward(features+3);
    // convolution_backward(features+4, *lenet);
    // dotproduct_backward(features+5, *lenet);
}

extern void freeLenet(){
    for(int i=0; i<LAYERS; i++){
        
    }
}