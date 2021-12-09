#include "lenet.h"

// ----- Training ----- //
void updateWeight(const number factor, Weight *weightGradient, Weight *weight) {
    uint w, m, wSize, mSize;
    Matrix *auxMatrix, *auxGradMatrix;
    wSize = WEIGHT_SIZE(weight);
    for(w = 0; w < wSize; w++) {
        auxGradMatrix = WEIGHT_GETMATRIX1(weightGradient, w);
        auxMatrix = WEIGHT_GETMATRIX1(weight, w);
        mSize = MATRIX_SIZE(auxMatrix);
        for(m = 0; m < mSize; m++)
            MATRIX_VALUE1(auxMatrix, m) += factor * MATRIX_VALUE1(auxGradMatrix, m);
    }
}

void updateBias(const number factor, Array *biasGradient, Array *bias) {
    for(uint n = 0; n < bias->n; n++)
        ARRAY_VALUE(bias, n) += factor * ARRAY_VALUE(biasGradient, n);
}

void updateParameters(const number factor, LeNet **lenetGradient, LeNet **lenet) {
    for(uint8 i = 0; i < 4; i++) {
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
        //Main process
        loadInput(input[i], *features);
        forwardPropagation(lenet, features);
        softMax(features[6], labels[i], featuresGradient[6]);
        backwardPropagation(lenet, features, lenetGradient, featuresGradient);
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
uint8 predict(LeNet **lenet, uint8 *input) {
    Feature **features = FEATURES_INITIAL();
    loadInput(input, *features);
    forwardPropagation(lenet, features);
    uint8 result = getResult(features[6]);
    freeFeatures(&features);
    return result;
}

uint8 getResult(Feature *features) {
    uint8 om, result = 0;
    number max = -1.0;
    Matrix *output = FEATURE_GETMATRIX(features, 0);
    for(om = 0; om < output->m; om++) {
        if(MATRIX_VALUE1(output, om) > max) {
            max = MATRIX_VALUE1(output, om);
            result = om;
        }
    }
    return result;
}

// ----- Propagation ----- //

void forwardPropagation(LeNet5 *lenet, Features *features) {
    convolution_forward(features->input, lenet->weight0_1, lenet->bias0_1, features->layer1);
    subsampling_forward(features->layer1, features->layer2);
    convolution_forward(features->layer2, lenet->weight2_3, lenet->bias2_3, features->layer3);
    subsampling_forward(features->layer3, features->layer4);
    convolution_forward(features->layer4, lenet->weight4_5, lenet->bias4_5, features->layer5);
    dotproduct_forward (features->layer5, lenet->weight5_6, lenet->bias5_6, features->output);
}

void backwardPropagation(LeNet **lenet, Feature **features, LeNet **lenetGradient, Feature **featuresGradient) {
    dotproduct_backward (features[5], *lenet[3], featuresGradient + 6, lenetGradient[3]);
    convolution_backward(features[4], *lenet[2], featuresGradient + 5, lenetGradient[2]);
    subsampling_backward(features[3], featuresGradient + 4);
    convolution_backward(features[2], *lenet[1], featuresGradient + 3, lenetGradient[1]);
    subsampling_backward(features[1], featuresGradient + 2);
    convolution_backward(features[0], *lenet[0], featuresGradient + 1, lenetGradient[0]);
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
        std += val*val;
    }
    mean = mean/IMG_SIZE;
    std = sqrt(std/IMG_SIZE - mean*mean);
    //Normalize data and add padding
    for(in = 0; in < IMG_ROWS; in++)
    for(im = 0; im < IMG_COLS; im++)
        MATRIX_VALUE(inputMatrix, (in + 2), (im + 2)) = (input[in*IMG_COLS + im] - mean) / std;
}

// ----- Initial values ----- //
void setInitialValues(LeNet **lenet) {
    srand(time(0));
    initialValues(lenet[0],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    initialValues(lenet[1],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    initialValues(lenet[2],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    initialValues(lenet[3],  sqrt(6.0 / (LAYER5 + OUTPUT)));
}

void randInitialValues(LeNet *lenet) {
    uint n, m, i, matrixSize;
    Matrix *matrix;
    for(n = 0; n < lenet->weight->n; n++) {
        for(m = 0; m < lenet->weight->m; m++) {
            matrix = WEIGHT_GETMATRIX(lenet->weight, n, m);
            matrixSize = MATRIX_SIZE(matrix);
            for(i = 0; i < matrixSize; i++)
                MATRIX_VALUE1(matrix, i) = f32Rand(1);
        }
    }
}

void initialValues(LeNet *lenet, const number value) {
    uint n, m, i, matrixSize;
    Matrix *matrix;
    randInitialValues(lenet);
    for(n = 0; n < lenet->weight->n; n++) {
        for(m = 0; m < lenet->weight->m; m++) {
            matrix = WEIGHT_GETMATRIX(lenet->weight, n, m);
            matrixSize = MATRIX_SIZE(matrix);
            for(i = 0; i < matrixSize; i++)
                MATRIX_VALUE1(matrix, i) *= value;
        }
    }
}
