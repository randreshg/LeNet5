#pragma once
#include "global/global.h"
#include "mnist/mnist.h"
/* ----- CONSTANTS ----- */
#define LEARNING_RATE   0.5
#define LENGTH_KERNEL   5
//Features
#define LENGTH_FEATURE0 32
#define LENGTH_FEATURE1 (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2 (LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3 (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4 (LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5 (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)
//Layer
#define LAYERS          6
#define INPUT           1
#define LAYER1          6
#define LAYER2          6
#define LAYER3          16
#define LAYER4          16
#define LAYER5          120
#define OUTPUT          10

/* ----- DATA STRUCTURES ----- */
typedef struct {
    Weight *weight;
    Array *bias;
} LeNet;

typedef struct {
    //Weight matrix
    number weight0_1[INPUT] [LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    number weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    number weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    number weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];
    //Bias
    number bias0_1[LAYER1];
    number bias2_3[LAYER3];
    number bias4_5[LAYER5];
    number bias5_6[OUTPUT];
}LeNet5;

typedef struct {
    number input [INPUT] [LENGTH_FEATURE0][LENGTH_FEATURE0];
    number layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    number layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    number layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    number layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    number layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    number output[OUTPUT];
}Features;

/* ----- CONSTRUCTOR ----- */
extern LeNet *LENET(const uint n, const uint m, const uint wm_n, const uint wm_m, const uint bi_n);
extern void LENET_FREE(LeNet **lenet);

extern void freeLenet(LeNet ***lenet);
extern void freeFeatures(Feature ***features);
/* ----- FUNCTIONS ----- */
//Training
extern void updateWeight(const number factor, Weight *weightGradient, Weight *weight);
extern void updateBias(const number factor, Array *biasGradient, Array *bias);
extern void updateParameters(const number factor, LeNet **lenetGradient, LeNet **lenet);
extern void trainBatch(LeNet **lenet, uint8 input[][IMG_SIZE], uint8 *labels, const uint batchSize);
//Prediction
extern uint8 predict(LeNet **lenet, uint8 *input);
extern uint8 getResult(Feature *features);
//Propagation
extern void forwardPropagation(LeNet **lenet, Feature **features);
extern void backwardPropagation(LeNet **lenet, Feature **features, LeNet **lenetGradient, Feature **featuresGradient);
//Others
extern void loadInput(uint8 *input, Feature *features);
//Initial values
extern LeNet **LENET_INITIAL();
extern Feature **FEATURES_INITIAL();
extern void setInitialValues(LeNet **lenet);
extern void initialValues(LeNet *lenet, const number value);
extern void randInitialValues(LeNet *lenet);

/* ----- FORWARD ----- */
extern void activation_forward(Feature *output, Array *bias, number (*action)(number));
extern void convolute_forward(Matrix *input, Matrix *weight, Matrix *output );
extern void convolution_forward(Feature **input, LeNet lenet);
extern void subsampling_forward(Feature **input);
extern void dotproduct_forward(Feature **input, LeNet lenet);

/* ----- BACKWARD ----- */
extern void activation_backward(Feature *input, Feature *output, number (*action)(number));
extern void convolute_backward(Matrix *input, Matrix *weight, Matrix *output);
extern void convolution_backward(Feature *input, LeNet lenet, Feature **inputGradient, LeNet *lenetGradient);
extern void subsampling_backward(Feature *input, Feature **inputGradient);
extern void dotproduct_backward(Feature *input, LeNet lenet, Feature **inputGradient, LeNet *lenetGradient);

/* ----- OTHERS ----- */
extern void softMax(Feature *input, uint8 target, Feature *featureGradient);
extern number costFunction(Feature *input, uint8 target);
extern number ReLU(number x);
extern number ReLU_GRAD(number x);
#define f32Rand(a) (((float)rand()/(float)(RAND_MAX))*(2*a) - a);

