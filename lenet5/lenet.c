#include "lenet.h"

// ----- Constructor ----- //
LeNet *LENET(const uint n, const uint m, const uint wm_n, const uint wm_m){
    LeNet *le = (LeNet *)malloc(sizeof(LeNet));
    le->weight = WEIGHT(n, m, wm_n, wm_m);
    le->bias = ARRAY(m);
    //initialValues(le);
    return le;
}

// ----- Destructor ----- //
void freeLenet(LeNet ***lenet){
    LeNet **aux = *lenet;
    for(int i=0; i<4; i++){
        WEIGHT_FREE(&(aux[i]->weight));
        ARRAY_FREE(&(aux[i]->bias));
        free(aux[i]);
    }
    free(aux);
}

void freeFeatures(Feature ***features){
    Feature **aux = *features;
    for(int i=0; i<LAYERS+1; i++)
        FEATURE_FREE(aux+i);
    free(aux);
}

// ----- Others ----- //
uint8 predict(LeNet **lenet, uint8 *input, uint8 count)
{
    //Features initial values
    Feature **features = FEATURES_INITIAL();
    //Load input
    image_char2float(input, FEATURE_GETMATRIX(*features, 0)->p);
    //Forward propagation
    forwardPropagation(lenet, features);
    uint8 result = getResult(features[6], count);
    //Free features
    freeFeatures(&features);
    return result;
}

uint8 getResult(Feature *features, uint8 count){
    uint8 om, result=-1, max = -1;
    Matrix *output = FEATURE_GETMATRIX(features, 0);
    for(om=0; om<output->m;om++){
        //printf("--%u: %f \n", om, MATRIX_VALUE1(output, om));
        if(MATRIX_VALUE1(output, om) > max){
            max = MATRIX_VALUE1(output, om);
            result = om;
        }
    }
    return result;
}

// ----- Propagation ----- //
void forwardPropagation(LeNet **lenet, Feature **features){
    convolution_forward(features, *lenet[0]);
    subsampling_forward(features+1);
    convolution_forward(features+2, *lenet[1]);
    subsampling_forward(features+3);
    convolution_forward(features+4, *lenet[2]);
    dotproduct_forward(features+5, *lenet[3]);
    //printf("OK \n");
}

void backwardPropagation(LeNet **lenet, Feature **features, Feature **gradientFeatures, LeNet **gradientLenet){
    dotproduct_backward(features[6], *lenet[3], gradientFeatures+6, gradientLenet[3]);
    // subsampling_backward(features[5], gradientFeatures+5);
    // convolution_backward(features[4], *lenet[2], gradientFeatures+4, gradientLenet[2]);
    // subsampling_backward(features[3], gradientFeatures+3);
    // convolution_backward(features[2], *lenet[1], gradientFeatures+2, gradientLenet[1]);
    // dotproduct_backward(features[1], *lenet[0], gradientFeatures+1, gradientLenet[0]);
}

// ----- Initial values ----- //
LeNet **LENET_INITIAL(){
    LeNet **lenet = malloc(4*sizeof(LeNet *));
    lenet[0] = LENET(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[1] = LENET(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[2] = LENET(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL);
    lenet[3] = LENET(1, 1, LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5, OUTPUT);
    return lenet;
}

Feature **FEATURES_INITIAL(){
    Feature **features = malloc(7*sizeof(LeNet *));
    features[0] = FEATURE(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0);
    features[1] = FEATURE(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1);
    features[2] = FEATURE(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2);
    features[3] = FEATURE(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3);
    features[4] = FEATURE(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4);
    features[5] = FEATURE(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5);
    features[6] = FEATURE(1, 1, OUTPUT);
    return features;
}

void setInitialValues(LeNet **lenet){
    srand((unsigned int)time(NULL));
    randInitialValues(lenet[0]);
    randInitialValues(lenet[1]);
    randInitialValues(lenet[2]);
    randInitialValues(lenet[3]);
    initialValues(lenet[0],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    initialValues(lenet[1],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    initialValues(lenet[2],  sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    initialValues(lenet[3],  sqrt(6.0 / (LAYER5 + OUTPUT)));
}

void randInitialValues(LeNet *lenet){
    uint n, m, i, matrixSize;
    Matrix *matrix;
    
    for(n=0; n<lenet->weight->n; n++){
        for(m=0; m<lenet->weight->m; m++){
            matrix = WEIGHT_GETMATRIX(lenet->weight, n, m);
            matrixSize = MATRIX_SIZE(matrix);
            for(i=0; i<matrixSize; i++){
                MATRIX_VALUE1(matrix, i) = f32Rand(0.00001);
            }
        }
    }
}

void initialValues(LeNet *lenet, const number value){
    uint n, m, i, matrixSize;
    Matrix *matrix;
    
    for(n=0; n<lenet->weight->n; n++){
        for(m=0; m<lenet->weight->m; m++){
            matrix = WEIGHT_GETMATRIX(lenet->weight, n, m);
            matrixSize = MATRIX_SIZE(matrix);
            for(i=0; i<matrixSize; i++)
                MATRIX_VALUE1(matrix, i) *=value;
        }
    }
}
