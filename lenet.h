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
#define LAYERS 6
#define INPUT           1
#define LAYER1          6
#define LAYER2          6
#define LAYER3          16
#define LAYER4          16
#define LAYER5          120
#define OUTPUT          10


// typedef struct
// {
//     //Weights
//     double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
//     double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
//     double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
//     double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];
//     //Bias
//     double bias0_1[LAYER1];
//     double bias2_3[LAYER3];
//     double bias4_5[LAYER5];
//     double bias5_6[OUTPUT];

// }LeNet;
// typedef struct
// {
//     double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
//     double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
//     double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
//     double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
//     double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
//     double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
//     double output[OUTPUT];
// }Feature;
/* ----- DATA TYPES ----- */
typedef unsigned char uint8;
typedef uint8 image[28][28];
typedef float number;

/* ----- DATA STRUCTURES ----- */
typedef struct
{
    Weight *weight;
    number *bias;
} LeNet;

typedef struct
{
    uint8 n;
    number *p;
} Array;

typedef struct
{
    uint8 n, m;
    number *p;
} Matrix;

typedef struct
{
    uint8 n;
    Matrix *p;
} Feature;

typedef struct
{
    uint8 n, m;
    Matrix *p;
} Weight;


/* ----- CONSTRUCTORS ----- */
Array *ARRAY(uint n)
{
    Array *ar = (Array *)malloc(sizeof(Array));
    ar->n = n, ar->p = malloc(sizeof(number)*(n));
    return ar;
}

Matrix *MATRIX(uint n, uint m)
{
    Matrix *ma = (Matrix *)malloc(sizeof(Matrix));
    ma->n = n, ma->m = m;
    ma->p = malloc(sizeof(number)*(n)*(m));
    return ma;
}

Feature *FEATURE(uint n, uint fl)
{
    Feature *fe = (Feature *)malloc(sizeof(Feature));
    fe->n = n, fe->p = MATRIX(fl, fl);
    return fe;
}

Weight *WEIGHT(uint n, uint m)
{
    Weight *we = (Weight *)malloc(sizeof(Weight));
    we->n = n, we->m = m;
    we->p = malloc(sizeof(Matrix)*(n)*(m));
    return we;
}

LeNet *LENET(uint n, uint m){
    LeNet *le = (LeNet *)malloc(sizeof(LeNet));
    le->weight = WEIGHT(n, m);
    le->bias = ARRAY(m);
}

/* ----- FUNCTIONS ----- */
void forwardPropagation(LeNet *lenet, Feature *features);
void backwardPropagation(double ***data);
void training(double ***data);
uint8 predict(LeNet *lenet, image input, uint8 count);
int testing(LeNet *lenet, image *test_data, uint8 *test_label,int total_size);
void initial(LeNet *lenet);
uint8 getResult(Feature *features, uint8 count);
void initialValues(double ***data);

