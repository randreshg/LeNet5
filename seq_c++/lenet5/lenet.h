#pragma once
/* ----- DEPENDENCIES ----- */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
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

typedef unsigned char uint8;
typedef unsigned int uint;
typedef float number;

/* ----- DATA STRUCTURES ----- */
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
}LeNet;

typedef struct {
    number input [INPUT] [LENGTH_FEATURE0][LENGTH_FEATURE0];
    number layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    number layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    number layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    number layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    number layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    number output[OUTPUT];
}Features;

/* ----- FUNCTIONS ----- */
//Training
extern void updateLenet(const number factor, LeNet *inputLenet, LeNet *outputLenet);
extern void trainBatch(LeNet *lenet, uint8 input[][IMG_SIZE], uint8 *labels, const uint batchSize);
//Prediction
extern uint8 predict(LeNet *lenet, uint8 *input);
extern uint8 getResult(number output[OUTPUT]);
//Propagation
extern void forwardPropagation(LeNet *lenet, Features *features);
extern void backwardPropagation(LeNet *lenet, Features *features, LeNet *lenetGradient, Features *featuresGradient);
//Initial values
extern void setInitialValues(LeNet *lenet);

/* ----- FORWARD ----- */
template<size_t ON, size_t ON1, size_t OM1, size_t BN>
extern void activation_forward(number (&output)[ON][ON1][OM1], number (&bias)[BN]);
template<size_t IN, size_t IM, size_t WN, size_t WM, size_t ON, size_t OM>
extern void convolute_forward(number (&input)[IN][IM], number (&weight)[WN][WM], number (&output)[ON][OM]);
template<size_t IN, size_t IN1, size_t IM1, size_t WN, size_t WM, size_t WN1, size_t WM1, size_t BN, size_t ON, size_t ON1, size_t OM1>
extern void convolution_forward(number (&input)[IN][IN1][IM1], number (&weight)[WN][WM][WN1][WM1], number (&bias)[BN], number (&output)[ON][ON1][OM1]);
template<size_t IN, size_t IN1, size_t IM1, size_t ON, size_t ON1, size_t OM1>
extern void subsampling_forward(number (&input)[IN][IN1][IM1], number (&output)[ON][ON1][OM1]);
template<size_t IN, size_t IN1, size_t IM1, size_t WN, size_t WM, size_t BN, size_t ON>
extern void dotproduct_forward(number (&input)[IN][IN1][IM1], number (&weight)[WN][WM], number (&bias)[BN], number (&output)[ON]);

/* ----- BACKWARD ----- */
template<size_t IN, size_t IN1, size_t IM1, size_t ON, size_t ON1, size_t OM1>
extern void activation_backward(number (&input)[IN][IN1][IM1], number (&output)[ON][ON1][OM1]);
template<size_t IN, size_t IM, size_t WN, size_t WM, size_t ON, size_t OM>
extern void convolute_backward(number (&input)[IN][IM], number (&weight)[WN][WM], number (&output)[ON][OM]);
template<size_t IN, size_t IN1, size_t IM1, size_t IGN, size_t IGN1, size_t IGM1,
         size_t WN, size_t WM,  size_t WN1, size_t WM1, size_t WGN, size_t WGM, size_t WGN1, size_t WGM1, size_t BG1,
         size_t OGN, size_t OGN1, size_t OGM1>
extern void convolution_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN][IGN1][IGM1],
                                 number (&weight)[WN][WM][WN1][WM1], number (&weightGradient)[WGN][WGM][WGN1][WGM1], number (&biasGradient)[BG1],
                                 number (&outputGradient)[OGN][OGN1][OGM1]);
template<size_t IN, size_t IN1, size_t IM1, size_t IGN, size_t IGN1, size_t IGM1, size_t OGN, size_t OGN1, size_t OGM1>
extern void subsampling_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN][IGN1][IGM1], number (&outputGradient)[OGN][OGN1][OGM1]);
template<size_t IN, size_t IN1, size_t IM1, size_t IGN,
         size_t WN, size_t WM, size_t WGN, size_t WGM, size_t BG1,
         size_t OGN, size_t OGN1, size_t OGM1>
extern void dotproduct_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN],
                                number (&weight)[WN][WM], number (&weightGradient)[WGN][WGM], number (&biasGradient)[BG1],
                                number (&outputGradient)[OGN][OGN1][OGM1]);

/* ----- OTHERS ----- */
#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))
#define GETCOUNT(array)  (sizeof(array)/sizeof(number))
#define f32Rand(a) (((number)rand()/(number)(RAND_MAX))*(2*a) - a)
extern number ReLU(number x);
extern number ReLU_GRAD(number x);
extern void loadInput(uint8 *input, number output[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0]);
extern void softMax(number input[OUTPUT], uint8 target, number outputGradient[OUTPUT]);

/* ----- TEMPLATE ----- */
#include "backward.tpp"
#include "forward.tpp"