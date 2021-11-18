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
#define TRAIN_IMAGE "data/train-images-idx3-ubyte"
#define TRAIN_LABEL "data/train-labels-idx1-ubyte"
#define TEST_IMAGE "data/t10k-images-idx3-ubyte"
#define TEST_LABEL "data/t10k-labels-idx1-ubyte"

#define IMG_ROWS 28  // 28
#define IMG_COLS 28  // 28
#define IMG_SIZE 784 // IMG_ROWS*IMG_COLS
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1


extern void FlipLong(unsigned char * ptr);
extern void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char data_char[][arr_n], int info_arr[]);
extern void image_char2double(unsigned char data_image_char[IMG_SIZE], double data_image[IMG_SIZE]);
extern void images_char2double(int num_data, unsigned char data_image_char[][IMG_SIZE], double data_image[][IMG_SIZE]);
extern void label_char2char(int num_data, unsigned char data_label_char[][1], unsigned char data_label[]);
extern void load_mnist(unsigned char train_image[NUM_TRAIN][IMG_SIZE], unsigned char test_image[NUM_TEST][IMG_SIZE], unsigned char train_label[NUM_TRAIN], unsigned char test_label[NUM_TEST]);
extern void load_testData(unsigned char test_image[NUM_TEST][IMG_SIZE], unsigned char test_label[NUM_TEST]);
extern void load_trainingData(unsigned char train_image[NUM_TRAIN][IMG_SIZE], unsigned char train_label[NUM_TRAIN]);
extern void print_mnist_img(double data_image[IMG_SIZE]);
extern void print_mnist_pixel(double data_image[][IMG_SIZE], int num_data);
extern void print_mnist_label(int data_label[], int num_data, int train_label[NUM_TRAIN], int test_label[NUM_TEST]);
extern void save_image(int n, char name[], unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE]);
extern void save_mnist_pgm(double data_image[][IMG_SIZE], int index);
