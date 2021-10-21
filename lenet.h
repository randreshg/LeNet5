#pragma once
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
#define LENGTH_CHANNEL1 6
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
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];
    //Bias
    double bias0_1[LAYER1];
    double bias2_3[LAYER3];
    double bias4_5[LAYER5];
    double bias5_6[OUTPUT];

}LeNet;

/* ----- FUNCTIONS ----- */
void forward(float ***data);
void backward(float ***data);
void training(float ***data);
int predict(float ***data, int input);
void initialValues(float ***data);

