//gcc -o main main.c lenet5/lenet.c lenet5/backward.c lenet5/forward.c lenet5/others.c
#include "lenet5/lenet.h"


int save(LeNet *lenet, char filename[])
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 1;
    fwrite(lenet, sizeof(LeNet), 1, fp);
    fclose(fp);
    return 0;
}

int load(LeNet *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet), 1, fp);
    fclose(fp);
    return 0;
}

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

void training(){
    //Read training data
    static uint8 train_image[NUM_TRAIN][IMG_SIZE];
    static uint8 train_label[NUM_TRAIN];
    load_trainingData(train_image, train_label);
    printf("TEST: %d \n", train_label[10]);
}

void trainBatch(LeNet **lenet, uint8 *input[IMG_SIZE], uint8 *labels, uint batchSize)
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
        image_char2float(input[i], FEATURE_GETMATRIX(*features, 0)->p);
        //Forward propagaton
        forwardPropagation(lenet, features);
        //Cost function
        cost = costFunction(features[6], labels[i]);
        //softMax
        softMax(features[6], labels[i], featuresGradient[6]);
        //Backward
        //backward(lenet, &deltas, &errors, &features, relugrad);
        //Update weights

        //Free memory
        freeFeatures(&features);
    }
    
    // double k = ALPHA / batchSize;
    // FOREACH(i, GETCOUNT(LeNet5))
    //     ((double *)lenet)[i] += k * buffer[i];
}

int main()
{
    bool train = false;
    if(train)
        training();
    printf("OK \n");
    //Load test data
    static uint8 test_image[NUM_TEST][IMG_SIZE]; 
    static uint8 test_label[NUM_TEST];
    load_testData(test_image, test_label);
    //printf("-%u \n", *test_label);
    //Malloc 
    LeNet **lenet = LENET_INITIAL();
    //Process starts
    clock_t start = clock();
    uint rightPredictions = testing(lenet, test_image, test_label, 1);
    //Process ends
    //printf("Results: %d/%d\n", rightPredictions, NUM_TEST);
    printf("Time: %u\n", (unsigned)(clock() - start));
    //Free
    freeLenet(&lenet);
    
    return 0;
}

