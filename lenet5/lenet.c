#include "lenet.h"


void forwardPropagation(LeNet *lenet, Feature *features){
    printf("Convolution 1 \n");
    //Layer 1
    convolution(features, *lenet);
    printf("Subsampling 1 \n");
    subsampling((double ***)features->layer1, (double ***)features->layer2);
    printf("Convolution 2 \n");
    convolution((double ***)features->layer2, (double ****)lenet->weight2_3, lenet->bias2_3, (double ***)features->layer3);
    subsampling((double ***)features->layer3, (double ***)features->layer4);
    convolution((double ***)features->layer4, (double ****)lenet->weight4_5, lenet->bias4_5, (double ***)features->layer5);
    dotproduct((double ***)features->layer5, (double **)lenet->weight5_6, lenet->bias5_6, features->output);
}

static inline void load_input(Feature *features, image input)
{
    MALLOC_FEATURE(features);

    double *layer0[LENGTH_FEATURE0][LENGTH_FEATURE0] = features[0].pointer;
    const long sz = sizeof(image) / sizeof(**input);
    double mean = 0, std = 0;

    const int jsize = sizeof(image) / sizeof(*input), ksize = sizeof(*input) / sizeof(**input);
    int j, k;
    for(j=0; j<jsize; j++)
        for(k=0; k<ksize; k++){
            mean += input[j][k];
            std += input[j][k] * input[j][k];
        }
    mean /= sz;
    std = sqrt(std / sz - mean*mean);
    for(j=0; j<jsize; j++)
        for(k=0; k<ksize; k++){
            layer0[0][j + 2][k + 2] = (input[j][k] - mean) / std;
        }
}

uint8 predict(LeNet *lenet, image input, uint8 count)
{
    Feature features[ ] = {
        {NULL, INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0*LENGTH_FEATURE0},
        {NULL, LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1*LENGTH_FEATURE1},
        {NULL, LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2*LENGTH_FEATURE2},
        {NULL, LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3*LENGTH_FEATURE3},
        {NULL, LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4*LENGTH_FEATURE4},
        {NULL, LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5*LENGTH_FEATURE5},
        {NULL, OUTPUT, 0, 0}
    };

    load_input(&features, input);
    
    forwardPropagation(lenet, &features);
    return getResult(&features, count);
}

uint8 getResult(Feature *features, uint8 count)
{
    double *output = (double *)features->output; 
    const int outlen = GET_COUNT(features->output);
    uint8 result = 0;
    double maxvalue = *output;
    for (uint8 i = 1; i < count; ++i)
    {
        if (output[i] > maxvalue)
        {
            maxvalue = output[i];
            result = i;
        }
    }
    return result;
}

int testing(LeNet *lenet, image *test_data, uint8 *test_label,int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i)
    {
        uint8 l = test_label[i];
        int p = predict(lenet, test_data[i], 10);
        right += l == p;
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);
    }
    return right;
}

static double f64rand()
{
    static int randbit = 0;
    if (!randbit)
    {
        srand((unsigned)time(0));
        for (int i = RAND_MAX; i; i >>= 1, ++randbit);
    }
    unsigned long long lvalue = 0x4000000000000000L;
    int i = 52 - randbit;
    for (; i > 0; i -= randbit)
        lvalue |= (unsigned long long)rand() << i;
    lvalue |= (unsigned long long)rand() >> -i;
    return *(double *)&lvalue - 3;
}

void initial(LeNet *lenet)
{
    for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
    for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
    for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}
