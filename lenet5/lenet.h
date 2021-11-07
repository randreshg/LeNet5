#pragma once
#include "global.h"

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

/* ----- CONSTRUCTORS ----- */
LeNet *LENET(uint n, uint m){
    LeNet *le = (LeNet *)malloc(sizeof(LeNet));
    le->weight = WEIGHT(n, m);
    le->bias = ARRAY(m);
}

/* ----- FUNCTIONS ----- */
void forwardPropagation(LeNet *lenet, Feature *features);
void backwardPropagation(LeNet *lenet, Feature *features);


void initialValues(double ***data);
void initial(LeNet *lenet);
void training(double ***data);
uint8 predict(LeNet *lenet, image input, uint8 count);
int testing(LeNet *lenet, image *test_data, uint8 *test_label,int total_size);
uint8 getResult(Feature *features, uint8 count);

/* ----- FORWARD ----- */
void convolute_forward(Matrix *input, Matrix *weight, Array *bias , Matrix *output );
void convolution_forward(Feature *input, LeNet lenet);
void subsampling_forward(Feature *input);
void dotproduct_forward(Feature *input, LeNet lenet);

/* ----- BACKWARD ----- */
void convolute_backward(Matrix *input, Matrix *weight, Array *bias , Matrix *output );
void convolution_backward(Feature *input, LeNet lenet);
void subsampling_backward(Feature *input);
void dotproduct_backward(Feature *input, LeNet lenet);

/* ----- OTHERS ----- */
#define ReLU(x) (x>0? x: 0)
#define ReLU_GRAD(x) (x>0)
void  softMax(Feature *input, Array *target, Feature *gradient);
number costFunction(Feature *input, Array *target);
