//gcc -o main main.c lenet5/lenet.c lenet5/backward.c lenet5/forward.c lenet5/others.c
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "lenet5/lenet.h"
#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define LENET_FILE1 		"model1.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000

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

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data)*count, 1, fp_image);
    fread(label,count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}


int main()
{
    LeNet **lenet = malloc(LAYERS*sizeof(LeNet));
    lenet[0] = LENET(INPUT, LAYER1);
    lenet[1] = LENET(LAYER2, LAYER3);
    lenet[2] = LENET(LAYER4, LAYER5);
    lenet[3] = LENET(1, 1);
    printf("TEST: %d", lenet[0]->bias->n);
    // image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    // uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    // image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    // uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
    // if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
    // {
    //     printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
    //     free(train_data);
    //     free(train_label);
    //     system("pause");
    // }
    // if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    // {
    //     printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
    //     free(test_data);
    //     free(test_label);
    //     system("pause");
    // }
    


    // LeNet *lenet = (LeNet *)malloc(sizeof(LeNet));
    // if (load(lenet, LENET_FILE1))
    //     initial(lenet);

    // clock_t start = clock();
    // // int batches[] = { 300 };
    // // for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
    // //     training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);
    
    // int right = testing(lenet, test_data, test_label, 50);
    // printf("%d/%d\n", right, COUNT_TEST);
    // printf("Time:%u\n", (unsigned)(clock() - start));
    // //save(lenet, LENET_FILE1);
    // free(lenet);
    // free(train_data);
    // free(train_label);
    // free(test_data);
    // free(test_label);
    return 0;
}

