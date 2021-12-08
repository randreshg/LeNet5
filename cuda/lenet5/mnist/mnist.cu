/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C

Edited by: Rafael Herrera /2021
*/

#include "mnist.h"

void FlipLong(unsigned char * ptr)
{
    register unsigned char val;
    
    // Swap 1st and 4th bytes
    val = *(ptr);
    *(ptr) = *(ptr+3);
    *(ptr+3) = val;
    
    // Swap 2nd and 3rd bytes
    ptr += 1;
    val = *(ptr);
    *(ptr) = *(ptr+1);
    *(ptr+1) = val;
}

void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char data_char[][arr_n], int info_arr[])
{
    int i, j, k, fd;
    unsigned char *ptr;

    if ((fd = open(file_path, O_RDONLY)) == -1) {
        fprintf(stderr, "couldn't open image file \n");
        exit(-1);
    }
    read(fd, info_arr, len_info * sizeof(int));
    // read-in information about size of data
    for (i=0; i<len_info; i++) { 
        ptr = (unsigned char *)(info_arr + i);
        FlipLong(ptr);
        ptr = ptr + sizeof(int);
    }
    // read-in mnist numbers (pixels|labels)
    for (i=0; i<num_data; i++) {
        read(fd, data_char[i], arr_n * sizeof(unsigned char));
    }
    close(fd);
}


void image_char2float(unsigned char data_image_char[IMG_SIZE], float data_image[IMG_SIZE])
{
    unsigned int i;
    for (i=0; i < IMG_SIZE; i++)
        data_image[i] = (float)data_image_char[i] / 255.0;
}

void images_char2float(int num_data, unsigned char data_image_char[][IMG_SIZE], float data_image[][IMG_SIZE])
{
    int i, j;
    for (i=0; i<num_data; i++)
        for (j=0; j<IMG_SIZE; j++)
            data_image[i][j]  = (float)data_image_char[i][j] / 255.0;
}

void label_char2char(int num_data, unsigned char data_label_char[][1], unsigned char data_label[])
{
    int i;
    for (i=0; i<num_data; i++)
        data_label[i]  = data_label_char[i][0];
}


void load_mnist(unsigned char train_image[NUM_TRAIN][IMG_SIZE], unsigned char test_image[NUM_TEST][IMG_SIZE], unsigned char train_label[NUM_TRAIN], unsigned char test_label[NUM_TEST])
{
    load_testData(test_image, test_label);
    load_trainingData(train_image, train_label);
}

void load_testData(unsigned char test_image[NUM_TEST][IMG_SIZE], unsigned char test_label[NUM_TEST])
{
    //Test
    unsigned char test_label_char[NUM_TEST][1];//, test_image_char[NUM_TEST][IMG_SIZE];
    int info_label[LEN_INFO_LABEL], info_image[LEN_INFO_IMAGE];
    read_mnist_char(TEST_IMAGE, NUM_TEST, LEN_INFO_IMAGE, IMG_SIZE, test_image, info_image);
    //image_char2float(NUM_TEST, test_image_char, test_image);
    read_mnist_char(TEST_LABEL, NUM_TEST, LEN_INFO_LABEL, 1, test_label_char, info_label);
    label_char2char(NUM_TEST, test_label_char, test_label);
}

void load_trainingData(unsigned char train_image[NUM_TRAIN][IMG_SIZE], unsigned char train_label[NUM_TRAIN])
{
    //Train
    unsigned char train_label_char[NUM_TRAIN][1]; //train_image_char[NUM_TRAIN][IMG_SIZE]
    int info_label[LEN_INFO_LABEL], info_image[LEN_INFO_IMAGE];
    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, LEN_INFO_IMAGE, IMG_SIZE, train_image, info_image);
    //image_char2float(NUM_TRAIN, train_image_char, train_image);
    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, LEN_INFO_LABEL, 1, train_label_char, info_label);
    label_char2char(NUM_TRAIN, train_label_char, train_label);
}


void print_mnist_img(float data_image[IMG_SIZE])
{
    int j;
    for (j=0; j<IMG_SIZE; j++) {
        printf("%1.1f ", data_image[j]);
        if ((j+1) % 28 == 0) putchar('\n');
    }
    putchar('\n');
}

void print_mnist_pixel(float data_image[][IMG_SIZE], int num_data)
{
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


void print_mnist_label(int data_label[], int num_data, int train_label[NUM_TRAIN], int test_label[NUM_TEST])
{
    int i;
    if (num_data == NUM_TRAIN)
        for (i=0; i<num_data; i++)
            printf("train_label[%d]: %d\n", i, train_label[i]);
    else
        for (i=0; i<num_data; i++)
            printf("test_label[%d]: %d\n", i, test_label[i]);
}