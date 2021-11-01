#include "lenet.h"

void model(){
    
}

void forwardPropagation(LeNet *lenet, Feature *features){
    convolution_forward(features, *lenet);
    subsampling_forward(features+1);
    convolution_forward(features+2, *lenet);
    subsampling_forward(features+3);
    convolution_forward(features+4, *lenet);
    dotproduct_forward(features+5, *lenet);
}

void backwardPropagation(LeNet *lenet, Feature *features){
    convolution_backward(features, *lenet);
    subsampling_backward(features+1);
    convolution_backward(features+2, *lenet);
    subsampling_backward(features+3);
    convolution_backward(features+4, *lenet);
    dotproduct_backward(features+5, *lenet);
}
