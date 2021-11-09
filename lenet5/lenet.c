#include "lenet.h"

LeNet *LENET(uint n, uint m){
    LeNet *le = (LeNet *)malloc(sizeof(LeNet));
    le->weight = WEIGHT(n, m);
    le->bias = ARRAY(m);
    return le;
}

void model(){
    
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