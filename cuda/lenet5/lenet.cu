#include "lenet.h"

// ----- Training ----- //
void updateLenet(const number factor, LeNet *inputLenet, LeNet *outputLenet) {
    const uint lenetSize = GETCOUNT(LeNet);
    for(uint n = 0; n < lenetSize; n++)
        ((number *)(outputLenet))[n] += factor * ((number *)(inputLenet))[n];
}

void trainBatch(LeNet *lenet, uint8 input[][IMG_SIZE], uint8 *labels, const uint batchSize) {
    // //Aux variables
    // const number alpha = LEARNING_RATE / batchSize;
    // LeNet lenetGradient, batchBuffer = {0};
    // Features features, featuresGradient;
    // for (uint i = 0; i < batchSize; i++) {
    //     //Malloc memory
    //     lenetGradient = {0};
    //     features = {0}, featuresGradient = {0};
    //     //Main process
    //     loadInput(input[i], features.input);
    //     forwardPropagation(lenet, &features);
    //     softMax(features.output, labels[i], featuresGradient.output);
    //     backwardPropagation(lenet, &features, &lenetGradient, &featuresGradient);
    //     updateLenet(1, &lenetGradient, &batchBuffer);
    // }
    // updateLenet(alpha, &batchBuffer, lenet);
}

// ----- Prediction ----- //
uint8 getResults(number output[OUTPUT]) {
    uint8 on, result = 0;
    number max = -1.0;
    for(on = 0; on < OUTPUT; on++)
        if(output[on] > max)
            max = output[on], result = on;
    return result;
}
__host__ uint predict(LeNet *lenet, uint8 *input, uint8 *labels) {
    //Features variables
    const int FEATURES_SIZE = BATCH_PARALLEL * sizeof(Features);
    Features *d_features;
    cudaMalloc((void **)&d_features, FEATURES_SIZE);
    cudaMemset(d_features, 0, FEATURES_SIZE);
    //Results variables
    uint *h_results, *d_results;
    cudaMallocHost((void **)&h_results, sizeof(uint));
    cudaMalloc((void **)&d_results, sizeof(uint));
    cudaMemset(d_results, 0, sizeof(uint));
    //Input
    loadInput<<<BATCH_PARALLEL, dim3(IMG_ROWS, IMG_COLS)>>>(input, d_features);
    forwardPropagation<<<BATCH_BLOCKS, BATCH_THREADS>>>(lenet, d_features);
    getResult<<<BATCH_BLOCKS, BATCH_THREADS>>>(d_features, labels, d_results);
    cudaDeviceSynchronize();
    //Copy results from device
    cudaMemcpy(h_results, d_results, sizeof(uint), cudaMemcpyDeviceToHost);
    uint results = *h_results;
    //Memory free
    cudaFree(d_features);
    cudaFree(h_results);  cudaFree(d_results);
    return results;
}


__global__ void getResult(Features *features, uint8 *labels, uint *results) {
    uint bn = blockIdx.x, tn = threadIdx.x;
    uint tid = bn*BATCH_THREADS + tn;
    Features feature = features[tid];
    uint8 on, result = 0;
    number max = -1.0;
    for(on = 0; on < OUTPUT; on++)
        if(feature.output[on] > max)
            max = feature.output[on], result = on;
    if(result == labels[tid])
        atomicAdd(results, 1);
}

// ----- Propagation ----- //
__global__ void forwardPropagation(LeNet *lenet, Features *features) {
    uint bn = blockIdx.x, tn = threadIdx.x;
    uint tid = bn*BATCH_THREADS + tn;
    Features *feature = &(features[tid]);
    convolution_forward<<<dim3(INPUT, LAYER1), dim3(LENGTH_FEATURE0, LENGTH_FEATURE0)>>>(feature->input, lenet->weight0_1, feature->layer1);
    activation_forward <<<LAYER1, LENGTH_FEATURE1*LENGTH_FEATURE1>>> (feature->layer1, lenet->bias0_1);
    subsampling_forward<<<LAYER2, dim3(LENGTH_FEATURE1, LENGTH_FEATURE1)>>> (feature->layer1, feature->layer2);
    convolution_forward<<<dim3(LAYER2, LAYER3), dim3(LENGTH_FEATURE2, LENGTH_FEATURE2)>>>(feature->layer2, lenet->weight2_3, feature->layer3);
    activation_forward <<<LAYER3, LENGTH_FEATURE3*LENGTH_FEATURE3>>> (feature->layer3, lenet->bias2_3);
    subsampling_forward<<<LAYER4, dim3(LENGTH_FEATURE3, LENGTH_FEATURE3)>>> (feature->layer3, feature->layer4);
    convolution_forward<<<dim3(LAYER4, LAYER5), dim3(LENGTH_FEATURE4, LENGTH_FEATURE4)>>>(feature->layer4, lenet->weight4_5, feature->layer5);
    activation_forward <<<LAYER5, LENGTH_FEATURE5*LENGTH_FEATURE5>>> (feature->layer5, lenet->bias4_5);
    dotproduct_forward <<<1, dim3((LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5)/2, OUTPUT)>>>(feature->layer5, lenet->weight5_6, lenet->bias5_6, feature->output);
}

__host__ void backwardPropagation(LeNet *lenet, Features *features, LeNet *lenetGradient, Features *featuresGradient) {
    // dotproduct_backward (features->layer5, featuresGradient->output, lenet->weight5_6, lenetGradient->weight5_6, lenetGradient->bias5_6, featuresGradient->layer5);
    // convolution_backward(features->layer4, featuresGradient->layer5, lenet->weight4_5, lenetGradient->weight4_5, lenetGradient->bias4_5, featuresGradient->layer4);
    // subsampling_backward(features->layer3, featuresGradient->layer4, featuresGradient->layer3);
    // convolution_backward(features->layer2, featuresGradient->layer3, lenet->weight2_3, lenetGradient->weight2_3, lenetGradient->bias2_3, featuresGradient->layer2);
    // subsampling_backward(features->layer1, featuresGradient->layer2, featuresGradient->layer1);
    // convolution_backward(features->input,  featuresGradient->layer1, lenet->weight0_1, lenetGradient->weight0_1, lenetGradient->bias0_1, featuresGradient->input);
}

// ----- Initial values ----- //
__host__ void setInitialValues(LeNet *lenet) {
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
