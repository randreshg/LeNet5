/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C

Edited by: Rafael Herrera /2021
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// set appropriate path for data
#define TRAIN_IMAGE "./data/train-images.idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels.idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images.idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels.idx1-ubyte"

#define IMG_SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

int width[MAX_NUM_OF_IMAGES], height[MAX_NUM_OF_IMAGES];


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
        fprintf(stderr, "couldn't open image file");
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


void image_char2float(int num_data, unsigned char data_image_char[][IMG_SIZE], float data_image[][IMG_SIZE])
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


void load_mnist(float train_image[NUM_TRAIN][IMG_SIZE], float test_image[NUM_TEST][IMG_SIZE], unsigned char train_label[NUM_TRAIN], unsigned char test_label[NUM_TEST])
{
    load_testData(test_image, test_label);
    load_trainingData(train_image, train_label);
}

void load_testData(float test_image[NUM_TEST][IMG_SIZE], unsigned char test_label[NUM_TEST])
{
    //Test
    unsigned char test_image_char[NUM_TEST][IMG_SIZE], test_label_char[NUM_TEST][1];
    int info_label[LEN_INFO_LABEL], info_image[LEN_INFO_IMAGE];
    read_mnist_char(TEST_IMAGE, NUM_TEST, LEN_INFO_IMAGE, IMG_SIZE, test_image_char, info_image);
    image_char2float(NUM_TEST, test_image_char, test_image);
    read_mnist_char(TEST_LABEL, NUM_TEST, LEN_INFO_LABEL, 1, test_label_char, info_label);
    label_char2char(NUM_TEST, test_label_char, test_label);
}

void load_trainingData(float train_image[NUM_TRAIN][IMG_SIZE], unsigned char train_label[NUM_TRAIN])
{
    //Train
    unsigned char train_image_char[NUM_TRAIN][IMG_SIZE], train_label_char[NUM_TRAIN][1];
    int info_label[LEN_INFO_LABEL], info_image[LEN_INFO_IMAGE];
    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, LEN_INFO_IMAGE, IMG_SIZE, train_image_char, info_image);
    image_char2float(NUM_TRAIN, train_image_char, train_image);
    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, LEN_INFO_LABEL, 1, train_label_char, info_label);
    label_char2char(NUM_TRAIN, train_label_char, train_label);
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


// name: path for saving image (ex: "./images/sample.pgm")
void save_image(int n, char name[], unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE])
{
    char file_name[MAX_FILENAME];
    FILE *fp;
    int x, y;

    if (name[0] == '\0') {
        printf("output file name (*.pgm) : ");
        scanf("%s", file_name);
    } else strcpy(file_name, name);

    if ( (fp=fopen(file_name, "wb"))==NULL ) {
        printf("could not open file\n");
        exit(1);
    }

    fputs("P5\n", fp);
    fputs("# Created by Image Processing\n", fp);
    fprintf(fp, "%d %d\n", width[n], height[n]);
    fprintf(fp, "%d\n", MAX_BRIGHTNESS);
    for (y=0; y<height[n]; y++)
        for (x=0; x<width[n]; x++)
            fputc(image[n][x][y], fp);
    fclose(fp);
    printf("Image was saved successfully\n");
}


// save mnist image (call for each image)
// store train_image[][] into image[][][]
void save_mnist_pgm(float data_image[][IMG_SIZE], int index)
{
    unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];
    int n = 0; // id for image (set to 0)
    int x, y;

    width[n] = 28;
    height[n] = 28;

    for (y=0; y<height[n]; y++) {
        for (x=0; x<width[n]; x++) {
            image[n][x][y] = data_image[index][y * width[n] + x] * 255.0;
        }
    }

    save_image(n, "", image);
}
