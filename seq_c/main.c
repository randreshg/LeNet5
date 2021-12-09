//gcc -o main main.c lenet5/lenet.c lenet5/backward.c lenet5/forward.c lenet5/others.c
#include "lenet5/lenet.h"

uint testing(LeNet **lenet, uint8 testImage[][IMG_SIZE], uint8 *testLabel, uint totalSize) {
    printf("--------\n");
    printf("TESTING\n");
    uint rightPredictions = 0, percent = 0, i;
    uint8 prediction;
    for (i = 0; i < totalSize; i++) {
        prediction = predict(lenet, testImage[i]);
        rightPredictions += (testLabel[i] == prediction);
        if (i * 100 / totalSize > percent)
            printf("test:%2d%%\n", percent = i*100/totalSize);
    }
    return rightPredictions;
}

void training(LeNet **lenet, const uint batchSize, const uint totalSize) {
    printf("--------\n");
    printf("TRAINING\n");
    setInitialValues(lenet);
    static uint8 trainImage[NUM_TRAIN][IMG_SIZE];
    static uint8 trainLabel[NUM_TRAIN];
    load_trainingData(trainImage, trainLabel);
    for (uint i = 0, percent = 0; i < totalSize; i += batchSize) {
        trainBatch(lenet, trainImage + i, trainLabel + i, batchSize);
        if (i * 100 / totalSize > percent)
            printf("Train:%2d%%\n", percent = i * 100 / totalSize);
    }
}

int main() {
    //Malloc 
    LeNet **lenet = LENET_INITIAL();
    printf("-------------------\n");
    printf("PROCESS STARTED\n ");
    //Training
    bool train = true;
    //Testing
    static uint8 testImage[NUM_TEST][IMG_SIZE]; 
    static uint8 testLabel[NUM_TEST];
    load_testData(testImage, testLabel);
    //Process starts
    clock_t start = clock();
    if(train)
        training(lenet, 300, NUM_TRAIN);
    else
        setInitialValues(lenet);
    uint rightPredictions = testing(lenet, testImage, testLabel, NUM_TEST);
    //Process ends
    printf("Results: %d/%d\n", rightPredictions, NUM_TEST);
    printf("Time: %u\n", (unsigned)(clock() - start));
    //Free
    printf("-------------------\n");
    printf("FREE LENET MEMORY\n");
    freeLenet(&lenet);
    return 0;
}

