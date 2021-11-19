#include "lenet.h"

// ----- Constructor ----- //
LeNet *LENET(const uint n, const uint m, const uint wm_n, const uint wm_m){
    LeNet *le = (LeNet *)malloc(sizeof(LeNet));
    le->weight = WEIGHT(n, m, wm_n, wm_m);
    le->bias = ARRAY(m);
    return le;
}

void LENET_FREE(LeNet **lenet){
    WEIGHT_FREE(&((*lenet)->weight));
    ARRAY_FREE(&((*lenet)->bias));
    free(*lenet);
    *lenet = NULL;
}

// ----- Destructors ----- //
void freeLenet(LeNet ***lenet){
    LeNet **aux = *lenet;
    for(int i = 0; i < 4; i++)
        LENET_FREE(aux + i);
    free(aux);
    aux = NULL;
}

void freeFeatures(Feature ***features){
    Feature **aux = *features;
    for(int i = 0; i < 7; i++)
        FEATURE_FREE(aux + i);
    free(aux);
    aux = NULL;
}

// ----- Training ----- //
void updateWeight(const number factor, Weight *weightGradient, Weight *weight) {
    uint w, m, wSize, mSize;
    Matrix *auxMatrix, *auxGradMatrix;
    wSize = WEIGHT_SIZE(weight);
    for(w = 0; w < wSize; w++){
        auxMatrix = WEIGHT_GETMATRIX1(weight, w);
        auxGradMatrix = WEIGHT_GETMATRIX1(weightGradient, w);
        mSize = MATRIX_SIZE(auxMatrix);
        for(m = 0; m < mSize; m++)
            MATRIX_VALUE1(auxMatrix, m) += factor * MATRIX_VALUE1(auxGradMatrix, m);
    }
}

void updateBias(const number factor, Array *biasGradient, Array *bias) {
    uint n;
    for(n = 0; n < bias->n; n++)
        ARRAY_VALUE(bias, n) += factor * ARRAY_VALUE(biasGradient, n);
}

void updateParameters(const number factor, LeNet **lenetGradient, LeNet **lenet) {
    uint8 i;
    for(i = 0; i < 4; i++){
        updateWeight(factor, lenetGradient[i]->weight, lenet[i]->weight);
        updateBias(factor, lenetGradient[i]->bias, lenet[i]->bias);
    }
}

void trainBatch(LeNet **lenet, uint8 input[][IMG_SIZE], uint8 *labels, const uint batchSize) {
    //Aux variables
    const number alpha = LEARNING_RATE / batchSize;
    LeNet **lenetGradient, **buffer = LENET_INITIAL();
    Feature **features, **featuresGradient;
    for (uint i = 0; i < batchSize; i++) {
        //Malloc memory
        lenetGradient = LENET_INITIAL();
        features = FEATURES_INITIAL();
        featuresGradient = FEATURES_INITIAL();
        //Load input
        loadInput(input[i], *features);
        //Forward propagation
        forwardPropagation(lenet, features);
        //SoftMax
        softMax(features[6], labels[i], featuresGradient[6]);
        //Backward
        backwardPropagation(lenet, features, lenetGradient, featuresGradient);
        //Update parameters
        updateParameters(1, lenetGradient, buffer);
        //Free memory
        freeLenet(&lenetGradient);
        freeFeatures(&features);
        freeFeatures(&featuresGradient);
    }
    updateParameters(alpha, buffer, lenet);
    freeLenet(&buffer);
}

// ----- Prediction ----- //
uint8 predict(LeNet **lenet, uint8 *input, uint8 count) {
    //Features initial values
    Feature **features = FEATURES_INITIAL();
    //Load input
    loadInput(input, *features);
    //Forward propagation
    forwardPropagation(lenet, features);
    uint8 result = getResult(features[6], count);
    //Free features
    freeFeatures(&features);
    return result;
}

uint8 getResult(Feature *features, uint8 count) {
    uint8 om, result=0;
    number max = -1.0;
    Matrix *output = FEATURE_GETMATRIX(features, 0);
    for(om = 0; om < output->m; om++){
        if(MATRIX_VALUE1(output, om) > max){
            max = MATRIX_VALUE1(output, om);
            result = om;
        }
    }
    return result;
}

// ----- Propagation ----- //
void forwardPropagation(LeNet **lenet, Feature **features) {
    convolution_forward(features, *lenet[0]);
    subsampling_forward(features+1);
    convolution_forward(features+2, *lenet[1]);
    subsampling_forward(features+3);
    convolution_forward(features+4, *lenet[2]);
    dotproduct_forward(features+5, *lenet[3]);
}

void backwardPropagation(LeNet **lenet, Feature **features, LeNet **lenetGradient, Feature **featuresGradient){
    dotproduct_backward(features[5], *lenet[3], featuresGradient+6, lenetGradient[3]);
    convolution_backward(features[4], *lenet[2], featuresGradient+5, lenetGradient[2]);
    subsampling_backward(features[3], featuresGradient+4);
    convolution_backward(features[2], *lenet[1], featuresGradient+3, lenetGradient[1]);
    subsampling_backward(features[1], featuresGradient+2);
    convolution_backward(features[0], *lenet[0], featuresGradient+1, lenetGradient[0]);
}

// ----- Others ----- //
void loadInput(uint8 *input, Feature *features) {
    //Aux variables
    Matrix *inputMatrix = FEATURE_GETMATRIX(features, 0);
    uint in, im;
    number mean = 0, std = 0, val;
    //Calculate standar deviation and mean
    for(in = 0; in < IMG_SIZE; in++){
        val = input[in];
        mean += val;
        std += val * val;
    }
    mean = mean / IMG_SIZE;
    std = sqrt(std / IMG_SIZE - mean * mean);
    //Normalize data and add padding
    for(in = 0; in < IMG_ROWS; in++)
    for(im = 0; im < IMG_COLS; im++)
        MATRIX_VALUE(inputMatrix, in + 2, im + 2) = (input[in*IMG_COLS + im] - mean) / std;
}

// ----- Initial values ----- //
LeNet **LENET_INITIAL() {
    LeNet **lenet = (LeNet **) malloc(4 * sizeof(LeNet *));
    lenet[0] = LENET(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[1] = LENET(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[2] = LENET(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[3] = LENET(1, 1, LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, OUTPUT);
    return lenet;
}

Feature **FEATURES_INITIAL() {
    Feature **features = (Feature **) malloc(7 * sizeof(Feature *));
    features[0] = FEATURE(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0);
    features[1] = FEATURE(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1);
    features[2] = FEATURE(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2);
    features[3] = FEATURE(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3);
    features[4] = FEATURE(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4);
    features[5] = FEATURE(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5);
    features[6] = FEATURE(1, 1, OUTPUT);
    return features;
}

void setInitialValues(LeNet **lenet) {
    initialValues(lenet[0],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    initialValues(lenet[1],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    initialValues(lenet[2],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    initialValues(lenet[3],  sqrt(6.0 / (LAYER5 + OUTPUT)));
}

static double f64rand()
{
    static int randbit = 0;
    if (!randbit)
    {
        srand((unsigned)time(0));
        for (int i = RAND_MAX; i; i >>= 1, ++randbit);
    }
    unsigned long long lvalue = 0x4000000000000000L;
    int i = 52 - randbit;
    for (; i > 0; i -= randbit)
        lvalue |= (unsigned long long)rand() << i;
    lvalue |= (unsigned long long)rand() >> -i;
    return *(double *)&lvalue - 3;
}

void randInitialValues(LeNet *lenet) {
    uint n, m, i, matrixSize;
    Matrix *matrix;
    for(n = 0; n < lenet->weight->n; n++){
        for(m = 0; m < lenet->weight->m; m++){
            matrix = WEIGHT_GETMATRIX(lenet->weight, n, m);
            matrixSize = MATRIX_SIZE(matrix);
            for(i = 0; i < matrixSize; i++){
                MATRIX_VALUE1(matrix, i) = f64rand();
            }
        }
    }
}

void initialValues(LeNet *lenet, const number value) {
    uint n, m, i, matrixSize;
    Matrix *matrix;
    randInitialValues(lenet);
    for(n = 0; n < lenet->weight->n; n++){
        for(m = 0; m < lenet->weight->m; m++){
            matrix = WEIGHT_GETMATRIX(lenet->weight, n, m);
            matrixSize = MATRIX_SIZE(matrix);
            for(i = 0; i < matrixSize; i++)
                MATRIX_VALUE1(matrix, i) *= value;
        }
    }
}
