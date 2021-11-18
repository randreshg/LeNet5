//gcc -o main main.c lenet5/lenet.c lenet5/backward.c lenet5/forward.c lenet5/others.c
#include "lenet5/lenet.h"

uint testing(LeNet **lenet, uint8 test_image[][IMG_SIZE], uint8 *test_label, uint total_size)
{
    printf("--------\n");
    printf("TESTING\n");
    uint rightPredictions=0, percent=0, i;
    uint8 prediction;
    for (i=0; i<total_size; i++){
        prediction = predict(lenet, test_image[i], 10);
        printf("TARGET:%u - PREDICTION:%u\n", test_label[i], prediction);
        rightPredictions += (test_label[i] == prediction);
        // if (i * 100 / total_size > percent)
        //     printf("test:%2d%%\n", percent = i*100/total_size);
    }
    return rightPredictions;
}

void training(LeNet **lenet, const uint batchSize, const uint totalSize)
{
    printf("--------\n");
    printf("TRAINING\n");
    setInitialValues(lenet);
    static uint8 train_image[NUM_TRAIN][IMG_SIZE];
    static uint8 train_label[NUM_TRAIN];
    load_trainingData(train_image, train_label);
    for (uint i = 0, percent = 0; i <= totalSize; i += batchSize) {
        trainBatch(lenet, train_image + i, train_label + i, batchSize);
        if (i * 100 / totalSize > percent)
            printf("Train:%2d%%\n", percent = i * 100 / totalSize);
    }
}

int main()
{
    //Malloc 
    LeNet **lenet = LENET_INITIAL();
    printf("-------------------\n");
    printf("PROCESS STARTED\n");
    //Training
    bool train = true;
    //Testing
    static uint8 test_image[NUM_TEST][IMG_SIZE]; 
    static uint8 test_label[NUM_TEST];
    load_testData(test_image, test_label);

    //Process starts
    clock_t start = clock();
    if(train)
        training(lenet, 3, 3);
    uint rightPredictions = testing(lenet, test_image, test_label, 3);
    //Process ends
    printf("Results: %d/%d\n", rightPredictions, NUM_TEST);
    printf("Time: %u\n", (unsigned)(clock() - start));
    //Free
    printf("-------------------\n");
    printf("FREE LENET MEMORY\n");
    freeLenet(&lenet);
    //
    return 0;
}

