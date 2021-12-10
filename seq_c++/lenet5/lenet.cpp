#include "lenet.h"

// ----- Training ----- //
void updateLenet(const number factor, LeNet *inputLenet, LeNet *outputLenet) {
    const uint lenetSize = GETCOUNT(LeNet);
    for(uint n = 0; n < lenetSize; n++)
        ((number *)(outputLenet))[n] += factor * ((number *)(inputLenet))[n];
}

void trainBatch(LeNet *lenet, uint8 input[][IMG_SIZE], uint8 *labels, const uint batchSize) {
    //Aux variables
    const number alpha = LEARNING_RATE / batchSize;
    LeNet lenetGradient, buffer = {0};
    Features features, featuresGradient;
    for (uint i = 0; i < batchSize; i++) {
        //Malloc memory
        lenetGradient = {0};
        features = {0}, featuresGradient = {0};
        //Main process
        loadInput(input[i], features.input);
        forwardPropagation(lenet, &features);
        softMax(features.output, labels[i], featuresGradient.output);
        backwardPropagation(lenet, &features, &lenetGradient, &featuresGradient);
        updateLenet(1, &lenetGradient, &buffer);
    }
    updateLenet(alpha, &buffer, lenet);
}

// ----- Prediction ----- //
uint8 predict(LeNet *lenet, uint8 *input) {
    Features features = {0};
    loadInput(input, features.input);
    forwardPropagation(lenet, &features);
    uint8 result = getResult(features.output);
    return result;
}

uint8 getResult(number output[OUTPUT]) {
    uint8 on, result = 0;
    number max = -1.0;
    for(on = 0; on < OUTPUT; on++)
        if(output[on] > max)
            max = output[on], result = on;
    return result;
}

// ----- Propagation ----- //
void forwardPropagation(LeNet *lenet, Features *features) {
    convolution_forward(features->input, lenet->weight0_1, lenet->bias0_1, features->layer1);
    subsampling_forward(features->layer1, features->layer2);
    convolution_forward(features->layer2, lenet->weight2_3, lenet->bias2_3, features->layer3);
    subsampling_forward(features->layer3, features->layer4);
    convolution_forward(features->layer4, lenet->weight4_5, lenet->bias4_5, features->layer5);
    dotproduct_forward (features->layer5, lenet->weight5_6, lenet->bias5_6, features->output);
}

void backwardPropagation(LeNet *lenet, Features *features, LeNet *lenetGradient, Features *featuresGradient) {
    dotproduct_backward (features->layer5, featuresGradient->output, lenet->weight5_6, lenetGradient->weight5_6, lenetGradient->bias5_6, featuresGradient->layer5);
    convolution_backward(features->layer4, featuresGradient->layer5, lenet->weight4_5, lenetGradient->weight4_5, lenetGradient->bias4_5, featuresGradient->layer4);
    subsampling_backward(features->layer3, featuresGradient->layer4, featuresGradient->layer3);
    convolution_backward(features->layer2, featuresGradient->layer3, lenet->weight2_3, lenetGradient->weight2_3, lenetGradient->bias2_3, featuresGradient->layer2);
    subsampling_backward(features->layer1, featuresGradient->layer2, featuresGradient->layer1);
    convolution_backward(features->input,  featuresGradient->layer1, lenet->weight0_1, lenetGradient->weight0_1, lenetGradient->bias0_1, featuresGradient->input);
}

// ----- Initial values ----- //
void setInitialValues(LeNet *lenet) {
    srand(time(0));
    number *pos;
    //Assign randon numbers to all weight matrices
    for (pos = (number *)lenet->weight0_1; pos < (number *)lenet->bias0_1;   *pos++ = f32Rand(1));
    //Scale values based on matrix dimension
    for (pos = (number *)lenet->weight0_1; pos < (number *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    for (pos = (number *)lenet->weight2_3; pos < (number *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    for (pos = (number *)lenet->weight4_5; pos < (number *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    for (pos = (number *)lenet->weight5_6; pos < (number *)lenet->bias0_1;   *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
    //Set biases values to 0
    for (pos = (number *)lenet->bias0_1;  pos < (number *)(lenet + 1); *pos++ = 0);
}
