#pragma once
#include "global/global.h"

/* ----- CONSTANTS ----- */
#define LENGTH_KERNEL   5
//Features
#define LENGTH_FEATURE0 32
#define LENGTH_FEATURE1 (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2 (LENGTH_FEATURE1 / 2)
#define LENGTH_FEATURE3 (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4 (LENGTH_FEATURE3 / 2)
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
extern LeNet *LENET(uint n, uint m);

/* ----- FUNCTIONS ----- */
extern void forwardPropagation(LeNet *lenet, Feature *features);
extern void backwardPropagation(LeNet *lenet, Feature *features);

extern void initialValues(double ***data);
extern void initial(LeNet *lenet);
extern void training(double ***data);
extern uint8 predict(LeNet *lenet, image input, uint8 count);
extern int testing(LeNet *lenet, image *test_data, uint8 *test_label,int total_size);
extern uint8 getResult(Feature *features, uint8 count);

/* ----- FORWARD ----- */
extern void activation_forward(Feature *output, Array *bias, number (*action)(number));
extern void convolute_forward(Matrix *input, Matrix *weight, Matrix *output );
extern void convolution_forward(Feature *input, LeNet lenet);
extern void subsampling_forward(Feature *input);
extern void dotproduct_forward(Feature *input, LeNet lenet);

/* ----- BACKWARD ----- */
extern void activation_backward(Feature *output, number (*action)(number));
extern void convolute_backward(Matrix *input, Matrix *weight, Matrix *output);
extern void convolution_backward(Feature *input, LeNet lenet, Feature *inputGradient, LeNet gradientLenet);
extern void subsampling_backward(Feature *input, Feature *inputGradient);
extern void dotproduct_backward(Feature *input, LeNet lenet, Feature *inputGradient, LeNet gradientLenet);

/* ----- OTHERS ----- */
extern void softMax(Feature *input, Array *target, Feature *gradient);
extern number costFunction(Feature *input, Array *target);
extern number ReLU(number x);
extern number ReLU_GRAD(number x);
