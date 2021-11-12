//gcc -o main main.c lenet5/lenet.c lenet5/backward.c lenet5/forward.c lenet5/others.c
#include "lenet5/lenet.h"

uint testing(LeNet **lenet, uint8 test_image[][IMG_SIZE], uint8 *test_label, uint total_size)
{
    uint rightPredictions=0, percent=0, i;
    uint8 prediction;
    for (i=0; i<total_size; i++){
        prediction = predict(lenet, test_image[i], 10);
        rightPredictions += (test_label[i] == prediction);
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i*100/total_size);
    }
    return rightPredictions;
}

void loadInput(uint8 input[][IMG_SIZE], Feature *features)
{
    //Aux variables
    Matrix *inputMatrix = FEATURE_GETMATRIX(features, 0);
    uint in, im;
    number mean = 0, std = 0, val;
    //Calculate standart deviation and mean
    for(in = 0; in<IMG_ROWS; in++)
    for(im = 0; im<IMG_COLS; im++) {
        val = input[in][im];
        mean += val;
        std += val*val;
    }
    mean = mean/IMG_SIZE;
    std = sqrt(std/IMG_SIZE - mean*mean);
    //Normalize data and add padding
    for(in=0; in<IMG_ROWS; in++)
    for(im=0; im<IMG_COLS; im++)
        MATRIX_VALUE(inputMatrix, in+2, im+2) = (input[in][im]-mean)/std;
}

void trainBatch(LeNet **lenet, uint8 input[][IMG_SIZE], uint8 *labels, uint batchSize)
{
    //Aux variables
    uint i;
    number cost;
    Feature **features;
    Feature **featuresGradient = FEATURES_INITIAL();
    LeNet **gradientLenet = LENET_INITIAL();
    for (i = 0; i < batchSize; i++)
    {
        //Malloc features
        features = FEATURES_INITIAL();
        //Load input
        loadInput(input, *features);
        //Forward propagation
        forwardPropagation(lenet, features);
        //SoftMax
        softMax(features[6], labels[i], featuresGradient[6]);
        //Cost function
        cost = costFunction(featuresGradient[6], labels[i]);
        printf("Cost: %f \n", cost);
        
        //Backward
        backwardPropagation(lenet, features, featuresGradient, gradientLenet);
        //Update weights

        //Free memory
        freeFeatures(&features);
    }
    
    // double k = ALPHA / batchSize;
    // FOREACH(i, GETCOUNT(LeNet5))
    //     ((double *)lenet)[i] += k * buffer[i];
}

void training(LeNet **lenet){
    //Read training data
    static uint8 train_image[NUM_TRAIN][IMG_SIZE];
    static uint8 train_label[NUM_TRAIN];
    load_trainingData(train_image, train_label);
    trainBatch(lenet, train_image, train_label, 1);
    // for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
    // {
    //     TrainBatch(lenet, train_data + i, train_label + i, batch_size);
    //     if (i * 100 / total_size > percent)
    //         printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
    // }
}
int main()
{
    //Malloc 
    LeNet **lenet = LENET_INITIAL();
    setInitialValues(lenet);
    //Training
    bool train = false;
    if(train)
        training(lenet);
    //printf("OK \n");
    //Testing
    static uint8 test_image[NUM_TEST][IMG_SIZE]; 
    static uint8 test_label[NUM_TEST];
    load_testData(test_image, test_label);

    //Process starts
    clock_t start = clock();
    uint rightPredictions = testing(lenet, test_image, test_label, 100);
    //Process ends
    printf("Results: %d/%d\n", rightPredictions, NUM_TEST);
    printf("Time: %u\n", (unsigned)(clock() - start));
    //Free
    freeLenet(&lenet);
    
    return 0;
}

