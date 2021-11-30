#include "lenet.h"

// ----- Constructor ----- //
LeNet *LENET(const uint n, const uint m, const uint wm_n, const uint wm_m, const uint bi_n) {
    LeNet *le = (LeNet *)malloc(sizeof(LeNet));
    le->weight = WEIGHT(n, m, wm_n, wm_m);
    le->bias = ARRAY(bi_n);
    return le;
}

void LENET_FREE(LeNet **lenet) {
    WEIGHT_FREE(&((*lenet)->weight));
    ARRAY_FREE(&((*lenet)->bias));
    free(*lenet);
    *lenet = NULL;
}

// ----- Destructors ----- //
void freeLenet(LeNet ***lenet) {
    LeNet **aux = *lenet;
    for(uint8 i = 0; i < 4; i++)
        LENET_FREE(aux + i);
    free(aux);
    aux = NULL;
}

void freeFeatures(Feature ***features) {
    Feature **aux = *features;
    for(uint8 i = 0; i < 7; i++)
        FEATURE_FREE(aux + i);
    free(aux);
    aux = NULL;
}

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
    //Features initial values
    Feature **features = FEATURES_INITIAL();
    //Load input
    loadInput(input, *features);
    //Forward propagation
    forwardPropagation(lenet, features);
    uint8 result = getResult(features[6]);
    //Free features
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
void forwardPropagation(LeNet **lenet, Feature **features) {
    convolution_forward(features, *lenet[0]);
    subsampling_forward(features + 1);
    convolution_forward(features + 2, *lenet[1]);
    subsampling_forward(features + 3);
    convolution_forward(features + 4, *lenet[2]);
    dotproduct_forward (features + 5, *lenet[3]);
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
LeNet **LENET_INITIAL() {
    LeNet **lenet = (LeNet **) malloc(4 * sizeof(LeNet *));
    lenet[0] = LENET(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL, LAYER1);
    lenet[1] = LENET(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL, LAYER3);
    lenet[2] = LENET(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL, LAYER5);
    lenet[3] = LENET(1, 1, LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, OUTPUT, OUTPUT);
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

//-------------------------------------------------------
#define LENET_FILE 		"model.dat"
#define LENET_FILE1 	"model1.dat"
typedef struct LeNet5
{
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

    double bias0_1[LAYER1];
    double bias2_3[LAYER3];
    double bias4_5[LAYER5];
    double bias5_6[OUTPUT];

}LeNet5;
int load(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    printf("OK \n");
    return 0;
}

//-------------------------------------------------------
void setInitialValues(LeNet **lenet) {
    // LeNet5 *lenet5 = (LeNet5 *)malloc(sizeof(LeNet5));
    // load(lenet5, LENET_FILE1);
    // //Aux variables
    // uint wn, wm, on, om;
    // Matrix *auxMatrix;
    // Weight *weight;
    // //Copy weight0_1
    // weight = lenet[0]->weight;
    // for(wn = 0; wn < weight->n; wn++)
    // for(wm = 0; wm < weight->m; wm++) {
    //     auxMatrix = WEIGHT_GETMATRIX(weight, wn, wm);
    //     for(on = 0; on < auxMatrix->n; on++)
    //     for(om = 0; om < auxMatrix->m; om++)
    //         MATRIX_VALUE(auxMatrix, on, om) = lenet5->weight0_1[wn][wm][on][om];
    // }
    // //Copy weight2_3
    // weight = lenet[1]->weight;
    // for(wn = 0; wn < weight->n; wn++)
    // for(wm = 0; wm < weight->m; wm++) {
    //     auxMatrix = WEIGHT_GETMATRIX(weight, wn, wm);
    //     for(on = 0; on < auxMatrix->n; on++)
    //     for(om = 0; om < auxMatrix->m; om++)
    //         MATRIX_VALUE(auxMatrix, on, om) = lenet5->weight2_3[wn][wm][on][om];
    // }
    // //Copy weight4_5
    // weight = lenet[2]->weight;
    // for(wn = 0; wn < weight->n; wn++)
    // for(wm = 0; wm < weight->m; wm++) {
    //     auxMatrix = WEIGHT_GETMATRIX(weight, wn, wm);
    //     for(on = 0; on < auxMatrix->n; on++)
    //     for(om = 0; om < auxMatrix->m; om++)
    //         MATRIX_VALUE(auxMatrix, on, om) = lenet5->weight4_5[wn][wm][on][om];
    // }
    // //Copy weight5_6
    // weight = lenet[3]->weight;
    // auxMatrix = WEIGHT_GETMATRIX1(weight, 0);
    // for(on = 0; on < auxMatrix->n; on++)
    // for(om = 0; om < auxMatrix->m; om++)
    //     MATRIX_VALUE(auxMatrix, on, om) = lenet5->weight5_6[on][om];
    //free(lenet5);
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
