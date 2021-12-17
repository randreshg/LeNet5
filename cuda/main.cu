//srun ./lenet -N1 --gres=gpu:1
#include "lenet5/lenet.h"
#define LENET_FILE "modelfnt.dat"

__host__ uint testing(LeNet *lenet, uint8 testImage[][IMG_SIZE], uint8 *testLabel, uint totalSize) {
    printf("--------\n");
    printf("TESTING\n");
    //Aux variables
    const uint INPUT_SIZE = F_PARALLEL*IMG_SIZE*sizeof(uint8);
    const uint LABEL_SIZE = F_PARALLEL*sizeof(uint8);
    uint i, rightPredictions = 0;
    uint8 *d_input, *d_label;
    //Loop
    for (i = 0; i < totalSize; i+= F_PARALLEL) {
        //Copy input to device
        cudaMalloc((void **)&d_input, INPUT_SIZE);
        cudaMemcpy(d_input, testImage[i], INPUT_SIZE, cudaMemcpyHostToDevice);
        //Copy labels to device
        cudaMalloc((void **)&d_label, LABEL_SIZE);
        cudaMemcpy(d_label, &testLabel[i], LABEL_SIZE, cudaMemcpyHostToDevice);
        //Predict
        rightPredictions += predict(lenet, d_input, d_label);
        //Free
        cudaFree(d_input);
        cudaFree(d_label);
    }
    return rightPredictions;
}

__host__ void training(LeNet *lenet, uint8 trainImage[][IMG_SIZE], uint8 *trainLabel, const uint totalSize) {
    printf("--------\n");
    printf("TRAINING\n");
    //Aux variables
    const uint INPUT_SIZE = B_PARALLEL*IMG_SIZE*sizeof(uint8);
    const uint LABEL_SIZE = B_PARALLEL*sizeof(uint8);
    uint i;
    uint8 *d_input, *d_label;
    for (i = 0; i < totalSize; i += B_PARALLEL) {
        //Copy input to device
        cudaMalloc((void **)&d_input, INPUT_SIZE);
        cudaMemcpy(d_input, trainImage[i], INPUT_SIZE, cudaMemcpyHostToDevice);
        //Copy labels to device
        cudaMalloc((void **)&d_label, LABEL_SIZE);
        cudaMemcpy(d_label, &trainLabel[i], LABEL_SIZE, cudaMemcpyHostToDevice);
        //Train
        trainBatch(lenet, d_input, d_label);
        //Free
        cudaFree(d_input); cudaFree(d_label);
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
    bool train = true;
    //Testing
    static uint8 testImage[NUM_TEST][IMG_SIZE]; 
    static uint8 testLabel[NUM_TEST];
    load_testData(testImage, testLabel);
    //Process starts
    cudaEventRecord(start);
    if(train) {
        //Train data
        static uint8 trainImage[NUM_TRAIN][IMG_SIZE];
        static uint8 trainLabel[NUM_TRAIN];
        load_trainingData(trainImage, trainLabel);
        //Initial values
        //setInitialValues(lenet);
        load(h_lenet, d_lenet, (char *)LENET_FILE);
        training(d_lenet, trainImage, trainLabel, NUM_TRAIN);
    }
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
