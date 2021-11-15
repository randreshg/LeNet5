//gcc -o main main.c lenet5/lenet.c lenet5/backward.c lenet5/forward.c lenet5/others.c
#include "lenet5/lenet.h"

uint testing(LeNet **lenet, uint8 test_image[][IMG_SIZE], uint8 *test_label, uint total_size)
{
    uint rightPredictions=0, percent=0, i;
    uint8 prediction;
    for (i=0; i<total_size; i++){
        prediction = predict(lenet, test_image[i], 10);
        // printf("TARGET:%u - PREDICTION:%u\n", test_label[i], prediction);
        // printf("------------------\n");
        rightPredictions += (test_label[i] == prediction);
        // if (i * 100 / total_size > percent)
        //     printf("test:%2d%%\n", percent = i*100/total_size);
    }
    return rightPredictions;
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
        loadInput(input[i], *features);
        //Forward propagation
        forwardPropagation(lenet, features);
        //SoftMax
        softMax(features[6], labels[i], featuresGradient[6]);
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
    bool train = true;
    if(train)
        training(lenet);
    //printf("OK \n");
    //Testing
    static uint8 test_image[NUM_TEST][IMG_SIZE]; 
    static uint8 test_label[NUM_TEST];
    load_testData(test_image, test_label);

    //Process starts
    clock_t start = clock();
    //uint rightPredictions = testing(lenet, test_image, test_label, 10);
    //Process ends
    //printf("Results: %d/%d\n", rightPredictions, NUM_TEST);
    printf("Time: %u\n", (unsigned)(clock() - start));
    //Free
    freeLenet(&lenet);
    
    return 0;
}

