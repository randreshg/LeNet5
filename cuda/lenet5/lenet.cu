#include "lenet.h"

// ----- Training ----- //
__global__ void updateLenet(const uint max, const number factor, LeNet *inputLenet, LeNet *outputLenet) {
    uint bn = blockIdx.x, tn = threadIdx.x;
    uint tid = bn*(blockDim.x) + tn;
    if(tid < max)
        atomicAdd(&(((number *)outputLenet)[tid]), factor * (((number *)(inputLenet))[tid]));
}

__host__ void trainBatch(LeNet *lenet, uint8 *input, uint8 *labels) {
    //Aux variables
    const number alpha = LEARNING_RATE / B_PARALLEL;
    //Lenet
    const uint LENET_SIZE = B_PARALLEL * sizeof(LeNet);
    LeNet *lenetGradients;
    cudaMalloc((void **)&lenetGradients, LENET_SIZE); cudaMemset(lenetGradients, 0, LENET_SIZE);
    //Features variables
    const uint FEATURES_SIZE = B_PARALLEL * sizeof(Features);
    Features *features, *featuresGradients;
    cudaMalloc((void **)&features, FEATURES_SIZE); cudaMemset(features, 0, FEATURES_SIZE);
    cudaMalloc((void **)&featuresGradients, FEATURES_SIZE); cudaMemset(featuresGradients, 0, FEATURES_SIZE);
    //Lenet
    LeNet *batchBuffer;
    cudaMalloc((void **)&batchBuffer, sizeof(LeNet)); cudaMemset(batchBuffer, 0, sizeof(LeNet));
    //Train
    loadInput<<<B_PARALLEL, dim3(IMG_ROWS, IMG_COLS)>>>(input, features);
    forwardPropagation<<<B_BLOCKS, B_THREADS>>>(lenet, features);
    softMax<<<B_PARALLEL, OUTPUT>>>(features, labels, featuresGradients);
    backwardPropagation<<<B_BLOCKS, B_THREADS>>>(lenet, features, featuresGradients, lenetGradients, batchBuffer);
    //Accumulate buffer
    updateLenet<<<U_BLOCKS, U_THREADS>>> (GETCOUNT(LeNet), alpha, batchBuffer, lenet);
    cudaDeviceSynchronize();
    //Free
    cudaFree(features); cudaFree(featuresGradients);
    cudaFree(batchBuffer); cudaFree(lenetGradients);
}

// ----- Prediction ----- //
__host__ uint predict(LeNet *lenet, uint8 *input, uint8 *labels) {
    //Features variables
    const int FEATURES_SIZE = F_PARALLEL * sizeof(Features);
    Features *features;
    cudaMalloc((void **)&features, FEATURES_SIZE);
    cudaMemset(features, 0, FEATURES_SIZE);
    //Results variables
    uint *h_results, *d_results;
    cudaMallocHost((void **)&h_results, sizeof(uint));
    cudaMalloc((void **)&d_results, sizeof(uint));
    cudaMemset(d_results, 0, sizeof(uint));
    //Input
    loadInput<<<F_PARALLEL, dim3(IMG_ROWS, IMG_COLS)>>>(input, features);
    forwardPropagation<<<F_BLOCKS, F_THREADS>>>(lenet, features);
    getResult<<<F_BLOCKS, F_THREADS>>>(features, labels, d_results);
    cudaDeviceSynchronize();
    //Copy results from device
    cudaMemcpy(h_results, d_results, sizeof(uint), cudaMemcpyDeviceToHost);
    uint results = *h_results;
    //Memory free
    cudaFree(features);
    cudaFree(h_results);  cudaFree(d_results);
    return results;
}

__global__ void getResult(Features *features, uint8 *labels, uint *results) {
    uint bn = blockIdx.x, tn = threadIdx.x;
    uint tid = bn*F_THREADS + tn;
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
    uint tid = bn*blockDim.x + tn;
    Features *feature = &(features[tid]);
    convolution_forward<<<dim3(INPUT, LAYER1), dim3(LENGTH_FEATURE0, LENGTH_FEATURE0)>>>(feature->input, lenet->weight0_1, feature->layer1);
    activation_forward <<<LAYER1, LENGTH_FEATURE1*LENGTH_FEATURE1>>> (feature->layer1, lenet->bias0_1);
    subsampling_forward<<<LAYER2, dim3(LENGTH_FEATURE1, LENGTH_FEATURE1)>>> (feature->layer1, feature->layer2);
    convolution_forward<<<dim3(LAYER2, LAYER3), dim3(LENGTH_FEATURE2, LENGTH_FEATURE2)>>>(feature->layer2, lenet->weight2_3, feature->layer3);
    activation_forward <<<LAYER3, LENGTH_FEATURE3*LENGTH_FEATURE3>>> (feature->layer3, lenet->bias2_3);
    subsampling_forward<<<LAYER4, dim3(LENGTH_FEATURE3, LENGTH_FEATURE3)>>> (feature->layer3, feature->layer4);
    convolution_forward<<<dim3(LAYER4, LAYER5), dim3(LENGTH_FEATURE4, LENGTH_FEATURE4)>>> (feature->layer4, lenet->weight4_5, feature->layer5);
    activation_forward <<<LAYER5, LENGTH_FEATURE5*LENGTH_FEATURE5>>> (feature->layer5, lenet->bias4_5);
    dotproduct_forward <<<1, dim3((LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5)/2, OUTPUT)>>>(feature->layer5, lenet->weight5_6, lenet->bias5_6, feature->output);
}

__global__ void backwardPropagation(LeNet *lenet, Features *features, Features *featuresGradients, LeNet *lenetGradients, LeNet *batchBuffer) {
    uint bn = blockIdx.x, tn = threadIdx.x;
    uint tid = bn*blockDim.x + tn;
    Features *feature = &(features[tid]);
    Features *featuresGradient = &(featuresGradients[tid]);
    LeNet *lenetGradient = &(lenetGradients[tid]);
    //Back propagation
    dotproduct_backward <<<1, dim3((LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5)/2, OUTPUT)>>>
        (feature->layer5, featuresGradient->output, lenet->weight5_6, lenetGradient->weight5_6, lenetGradient->bias5_6, featuresGradient->layer5);
    convolution_backward<<<dim3(LAYER4, LAYER5), dim3(LENGTH_FEATURE4, LENGTH_FEATURE4)>>>
        (feature->layer4, featuresGradient->layer5, lenet->weight4_5, lenetGradient->weight4_5, lenetGradient->bias4_5, featuresGradient->layer4);
    activation_backward <<<LAYER4, LENGTH_FEATURE4*LENGTH_FEATURE4>>> (feature->layer4, featuresGradient->layer4);
    subsampling_backward<<<LAYER4, dim3(LENGTH_FEATURE3, LENGTH_FEATURE3)>>>(feature->layer3, featuresGradient->layer4, featuresGradient->layer3);
    convolution_backward <<<dim3(LAYER2, LAYER3), dim3(LENGTH_FEATURE2, LENGTH_FEATURE2)>>>
        (feature->layer2, featuresGradient->layer3, lenet->weight2_3, lenetGradient->weight2_3, lenetGradient->bias2_3, featuresGradient->layer2);
    activation_backward <<<LAYER2, LENGTH_FEATURE2*LENGTH_FEATURE2>>> (feature->layer2, featuresGradient->layer2);
    subsampling_backward<<<LAYER2, dim3(LENGTH_FEATURE1, LENGTH_FEATURE1)>>>(feature->layer1, featuresGradient->layer2, featuresGradient->layer1);
    convolution_backward<<<dim3(INPUT, LAYER1), dim3(LENGTH_FEATURE0, LENGTH_FEATURE0)>>>
        (feature->input,  featuresGradient->layer1, lenet->weight0_1, lenetGradient->weight0_1, lenetGradient->bias0_1, featuresGradient->input);
    activation_backward <<<INPUT, LENGTH_FEATURE0*LENGTH_FEATURE0>>> (feature->input, featuresGradient->input);
    //Accumulate buffer
    updateLenet<<<U_BLOCKS, U_THREADS>>> (GETCOUNT(LeNet), 1.0, lenetGradient, batchBuffer);
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
