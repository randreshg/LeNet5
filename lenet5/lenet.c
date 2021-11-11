#include "lenet.h"

LeNet *LENET(const uint n, const uint m, const uint wm_n, const uint wm_m){
    LeNet *le = (LeNet *)malloc(sizeof(LeNet));
    le->weight = WEIGHT(n, m, wm_n, wm_m);
    le->bias = ARRAY(m);
    initialValues(le);
    return le;
}

void freeLenet(LeNet **lenet){
    for(int i=0; i<4; i++){
        WEIGHT_FREE(&(lenet[i]->weight));
        ARRAY_FREE(&(lenet[i]->bias));
        free(lenet[i]);
    }
}

void freeFeatures(Feature **features){
    for(int i=0; i<LAYERS+1; i++)
        FEATURE_FREE(features+i);
}

uint8 predict(LeNet **lenet, uint8 *input, uint8 count)
{
    //Malloc features
    Feature **features = malloc((LAYERS+1)*sizeof(Feature *));;
    FEATURES_INITIAL(features);
    //Load input
    image_char2float(input, FEATURE_GETMATRIX(*features, 0)->p);
    //Forward propagation
    forwardPropagation(lenet, features);
    //uint8 result = get_result(features[6], count);
    freeFeatures(features);
    return 1;
}

// ----- Propagation ----- //
void forwardPropagation(LeNet **lenet, Feature **features){
    convolution_forward(features, *lenet[0]);
    subsampling_forward(features+1);
    convolution_forward(features+2, *lenet[1]);
    subsampling_forward(features+3);
    convolution_forward(features+4, *lenet[2]);
    dotproduct_forward(features+5, *lenet[3]);
    printf("OK \n");
}

void backwardPropagation(LeNet *lenet, Feature *features){
    // convolution_backward(features, *lenet);
    // subsampling_backward(features+1);
    // convolution_backward(features+2, *lenet);
    // subsampling_backward(features+3);
    // convolution_backward(features+4, *lenet);
    // dotproduct_backward(features+5, *lenet);
}

// ----- Initial values ----- //
void initialValues(LeNet *lenet){
    uint n, m, i, matrixSize;
    Matrix *matrix;
    
    for(n=0; n<lenet->weight->n; n++){
        for(m=0; m<lenet->weight->m; m++){
            matrix = WEIGHT_GETMATRIX(lenet->weight, n, m);
            matrixSize = MATRIX_SIZE(matrix);
            for(i=0; i<matrixSize; i++){
                MATRIX_VALUE1(matrix, i) = f32Rand(10);
            }
        }
    }
}

void LENET_INITIAL(LeNet **lenet){
    srand((unsigned int)time(NULL));
    lenet[0] = LENET(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[1] = LENET(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[2] = LENET(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[3] = LENET(1, 1, LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5, OUTPUT);
}

void FEATURES_INITIAL(Feature **features){
    features[0] = FEATURE(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0);
    features[1] = FEATURE(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1);
    features[2] = FEATURE(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2);
    features[3] = FEATURE(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3);
    features[4] = FEATURE(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4);
    features[5] = FEATURE(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5);
    features[6] = FEATURE(1, 1, OUTPUT);
}
