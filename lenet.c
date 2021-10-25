#include "lenet.h"
#include <stdlib.h>


void forwardPropagation(LeNet *lenet, Feature *features){
    convolution(features->input, lenet->weight0_1, lenet->bias0_1, features->layer1);
    subsampling(features->layer1, features->layer2);
    convolution(features->layer2, lenet->weight2_3, lenet->bias2_3, features->layer3);
    subsampling(features->layer3, features->layer4);
    convolution(features->layer4, lenet->weight4_5, lenet->bias4_5, features->layer5);
    dotproduct(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6);
}

uint8 predict(LeNet *lenet, image input, uint8 count)
{
    Feature features = { 0 };
    load_input(&features, input);
    forward(lenet, &features);
    return get_result(&features, count);
}

static uint8 get_result(Feature *features, uint8 count)
{
    double *output = (double *)features->output; 
    const int outlen = GETCOUNT(features->output);
    uint8 result = 0;
    double maxvalue = *output;
    for (uint8 i = 1; i < count; ++i)
    {
        if (output[i] > maxvalue)
        {
            maxvalue = output[i];
            result = i;
        }
    }
    return result;
}