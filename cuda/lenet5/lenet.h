#pragma once
#include "global/global.h"
#include "mnist/mnist.h"
#include <cuda_runtime.h>
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
typedef struct
{
    Weight *weight;
    Array *bias;
} LeNet;

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
#define f32Rand(a) (((double)rand()/(double)(RAND_MAX))*(2*a) - a);


