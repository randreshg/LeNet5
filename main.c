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

uint testing(LeNet **lenet, float *test_data[IMG_SIZE], uint8 *test_label, uint total_size)
{
    uint rightPredictions=0, percent=0, i;
    uint8 prediction;
    for (i=0; i<total_size; i++){
        prediction = predict(lenet, test_data[i], 10);
        rightPredictions += (test_label[i] == prediction);
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i*100/total_size);
    }
    return rightPredictions;
}

int main()
{
    bool training = false;
    // if(training){
    //     //Read training data
    //     float train_image[NUM_TRAIN][IMG_SIZE];
    //     uint8 train_label[NUM_TRAIN];
    //     load_trainingData(train_image, train_label);
    // }
    // printf("OK \n");
    // //Read testing data
    
    //load_trainingData(test_image, test_label);

    //Malloc 
    LeNet **lenet = malloc(4*sizeof(LeNet *));;
    LENET_INITIAL(lenet);
    //Process starts
    clock_t start = clock();
    uint rightPredictions = testing(lenet, test_image, test_label, 1);
    //printf("Results: %d/%d\n", rightPredictions, NUM_TEST);
    //Process ends
    printf("Time: %u\n", (unsigned)(clock() - start));
    freeLenet(lenet);
    
    return 0;
}

