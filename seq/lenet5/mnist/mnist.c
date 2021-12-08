/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C

Edited by: Rafael Herrera /2021
*/

#include "mnist.h"

void read_data(const int count, const char data_file[], const char label_file[], unsigned char data[][IMG_SIZE], unsigned char label[]) {
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) {
        fprintf(stderr, "couldn't open image file \n");
        exit(-1);
    }
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data)*count, 1, fp_image);
    fread(label, count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
}

void image_char2float(unsigned char data_image_char[IMG_SIZE], float data_image[IMG_SIZE]) {
    unsigned int i;
    for (i=0; i < IMG_SIZE; i++)
        data_image[i] = (float)data_image_char[i];
}

void images_char2float(int num_data, unsigned char data_image_char[][IMG_SIZE], float data_image[][IMG_SIZE]) {
    int i, j;
    for (i=0; i<num_data; i++)
        for (j=0; j<IMG_SIZE; j++)
            data_image[i][j]  = (float)data_image_char[i][j];
}

void load_mnist(unsigned char train_image[NUM_TRAIN][IMG_SIZE], unsigned char test_image[NUM_TEST][IMG_SIZE], unsigned char train_label[NUM_TRAIN], unsigned char test_label[NUM_TEST]) {
    load_testData(test_image, test_label);
    load_trainingData(train_image, train_label);
}

void load_testData(unsigned char test_image[NUM_TEST][IMG_SIZE], unsigned char test_label[NUM_TEST]) {
    read_data(NUM_TEST, TEST_IMAGE, TEST_LABEL, test_image, test_label);
}

void load_trainingData(unsigned char train_image[NUM_TRAIN][IMG_SIZE], unsigned char train_label[NUM_TRAIN]) {
    read_data(NUM_TRAIN, TRAIN_IMAGE, TRAIN_LABEL, train_image, train_label);
}


void print_mnist_img(float data_image[IMG_SIZE]) {
    int j;
    for (j=0; j<IMG_SIZE; j++) {
        printf("%1.1f ", data_image[j]);
        if ((j+1) % 28 == 0) putchar('\n');
    }
    putchar('\n');
}

void print_mnist_pixel(float data_image[][IMG_SIZE], int num_data) {
    int i, j;
    for (i=0; i<num_data; i++) {
        printf("image %d/%d\n", i+1, num_data);
        for (j=0; j<IMG_SIZE; j++) {
            printf("%1.1f ", data_image[i][j]);
            if ((j+1) % 28 == 0) putchar('\n');
        }
        putchar('\n');
    }
}


void print_mnist_label(int data_label[], int num_data, int train_label[NUM_TRAIN], int test_label[NUM_TEST]) {
    int i;
    if (num_data == NUM_TRAIN)
        for (i=0; i<num_data; i++)
            printf("train_label[%d]: %d\n", i, train_label[i]);
    else
        for (i=0; i<num_data; i++)
            printf("test_label[%d]: %d\n", i, test_label[i]);
}