#include "lenet5/lenet.h"
#include <omp.h>
#define LENET_FILE "modelfnt.dat"

uint testing(LeNet *lenet, uint8 testImage[][IMG_SIZE], uint8 *testLabel, uint totalSize) {
    printf("--------\n");
    printf("TESTING\n");
    uint i, rightPredictions = 0;
    #pragma omp parallel for if(OPENMP)
    for (i = 0; i < totalSize; i++) {
        uint8 prediction = predict(lenet, testImage[i]);
        #pragma omp critical
        rightPredictions += (testLabel[i] == prediction);
    }
    return rightPredictions;
}

void training(LeNet *lenet, const uint batchSize, const uint totalSize) {
    printf("--------\n");
    printf("TRAINING\n");
    //Train data
    static uint8 trainImage[NUM_TRAIN][IMG_SIZE];
    static uint8 trainLabel[NUM_TRAIN];
    load_trainingData(trainImage, trainLabel);
    //Train loop
    uint i, aux, percent = 0;
    for (i = 0; i < totalSize; i += batchSize) {
        trainBatch(lenet, trainImage + i, trainLabel + i, batchSize);
        aux = i*100/totalSize;
        if (aux > percent)
            printf("Train:%2d%%\n", percent = aux);
    }
}

void load(LeNet *lenet, char filename[]) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Model not found \n");
        exit(0);
    }
    fread(lenet, sizeof(LeNet), 1, fp);
    fclose(fp);
}

void save(LeNet *lenet, char filename[]) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Model not found \n");
        exit(0);
    }
    fwrite(lenet, sizeof(LeNet), 1, fp);
	fclose(fp);
}

int main() {
    //Malloc 
    LeNet lenet;
    printf("-------------------\n");
    printf("PROCESS STARTED\n ");
    //Training
    bool train = true;
    //Testing
    static uint8 testImage[NUM_TEST][IMG_SIZE]; 
    static uint8 testLabel[NUM_TEST];
    load_testData(testImage, testLabel);
    //Process starts
    double itime, t_time, ftime, exec_time;
    itime = omp_get_wtime();
    if(train) {
        //setInitialValues(lenet);
        load(&lenet, (char *)LENET_FILE);
        training(&lenet, 300, NUM_TRAIN);
    }
    else
        load(&lenet, (char *)LENET_FILE);
    t_time = omp_get_wtime() - itime;
    uint rightPredictions = testing(&lenet, testImage, testLabel, NUM_TEST);
    //Process ends
    ftime = omp_get_wtime();
    exec_time = ftime - itime;
    //save(&lenet, (char *)LENET_FILE);
    printf("-------------------\n");
    printf("PROCESS FINISHED\n ");
    printf("Results: %d/%d\n", rightPredictions, NUM_TEST);
    printf("Training time (s): %f \n", t_time);
    printf("Execution time (s): %f \n", exec_time);
    return 0;
}

