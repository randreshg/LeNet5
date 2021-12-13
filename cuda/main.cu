//srun ./lenet -N1 --gres=gpu:1
#include "lenet5/lenet.h"
#define LENET_FILE "modelf.dat"

__host__ uint testing(LeNet *lenet, uint8 testImage[][IMG_SIZE], uint8 *testLabel, uint totalSize) {
    printf("--------\n");
    printf("TESTING\n");
    //Aux variables
    const uint INPUT_SIZE = BATCH_PARALLEL*IMG_SIZE*sizeof(uint8);
    const uint LABEL_SIZE = BATCH_PARALLEL*sizeof(uint8);
    uint i, rightPredictions = 0, percent = 0, aux;
    uint8 *d_input, *d_label;
    //Loop
    for (i = 0; i < totalSize; i+= BATCH_PARALLEL) {
        //Copy input to device
        cudaMalloc((void **)&d_input, INPUT_SIZE);
        cudaMemcpy(d_input, testImage[i], INPUT_SIZE, cudaMemcpyHostToDevice);
        //Copy labels to device
        cudaMalloc((void **)&d_label, LABEL_SIZE);
        cudaMemcpy(d_label, &testLabel[i], LABEL_SIZE, cudaMemcpyHostToDevice);
        //Predict
        rightPredictions += predict(lenet, d_input, d_label);
        // aux = i*100/totalSize;
        // if (aux > percent)
        //     printf("test:%2d%%\n", percent = aux);
        //Free
        cudaFree(d_input);
        cudaFree(d_label);
    }
    return rightPredictions;
}

__host__ void training(LeNet *lenet, const uint batchSize, const uint totalSize) {
    printf("--------\n");
    printf("TRAINING\n");
    setInitialValues(lenet);
    //Train data
    static uint8 trainImage[NUM_TRAIN][IMG_SIZE];
    static uint8 trainLabel[NUM_TRAIN];
    load_trainingData(trainImage, trainLabel);
    //Train data
    uint i, aux, percent = 0;
    for (i = 0; i < totalSize; i += batchSize) {
        trainBatch(lenet, trainImage + i, trainLabel + i, batchSize);
        aux = i*100/totalSize;
        if (aux > percent)
            printf("Train:%2d%%\n", percent = aux);
    }
}

__host__ void load(LeNet *h_lenet, LeNet *d_lenet, char filename[]) {
    //Read file
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Model not found \n");
        exit(0);
    }
    fread(h_lenet, sizeof(LeNet), 1, fp);
    fclose(fp);
    //Copy info to device
    cudaMemcpy(d_lenet, h_lenet, sizeof(LeNet), cudaMemcpyHostToDevice);
}

int main() {
    printf("-------------------\n");
    printf("PROCESS STARTED\n");
    //Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //Allocate memory
    LeNet *h_lenet, *d_lenet;
    cudaMallocHost((void **)&h_lenet, sizeof(LeNet));
    cudaMalloc((void **)&d_lenet, sizeof(LeNet));
    //Training
    bool train = false;
    //Testing
    static uint8 testImage[NUM_TEST][IMG_SIZE]; 
    static uint8 testLabel[NUM_TEST];
    load_testData(testImage, testLabel);
    //Process starts
    cudaEventRecord(start);
    if(train)
        training(d_lenet, 300, NUM_TRAIN);
    else
        load(h_lenet, d_lenet, (char *)LENET_FILE);
    uint rightPredictions = testing(d_lenet, testImage, testLabel, NUM_TEST);
    cudaEventRecord(stop);
    //Process ends
    printf("-------------------\n");
    printf("PROCESS FINISHED\n ");
    printf("Results: %d/%d\n", rightPredictions, NUM_TEST);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time (ms): %f\n", milliseconds);
    //Memory free
    cudaFree(h_lenet);
    cudaFree(d_lenet);
    cudaDeviceReset();
    return 0;
}

