#pragma once

typedef unsigned char uint8;
typedef uint8 image[28][28];

/* ----- RESULTS ----- */
#define Result ***float

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
#define LAYERS 6
#define INPUT           1
#define LAYER1          6
#define LAYER2          6
#define LAYER3          16
#define LAYER4          16
#define LAYER5          120
#define OUTPUT          10

/* ----- DATA STRUCTURE ----- */
typedef struct
{
    //Weights
    float weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    float weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    float weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    float weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];
    //Bias
    float bias0_1[LAYER1];
    float bias2_3[LAYER3];
    float bias4_5[LAYER5];
    float bias5_6[OUTPUT];

}LeNet;

typedef struct
{
    float input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    float layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    float layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    float layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    float layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    float layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    float output[OUTPUT];
}Feature;

/* ----- FUNCTIONS ----- */
void forwardPropagation(LeNet *lenet, Feature *features);
void backwardPropagation(float ***data);
void training(float ***data);
uint8 predict(LeNet *lenet, image input, uint8 count);

void initialValues(float ***data);

