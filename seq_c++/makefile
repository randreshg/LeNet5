MKDIR_P := mkdir -p
OUT_DIR := build
OBJDIR    =   objdir
OBJ = $(addprefix $(OBJDIR)/, $(patsubst %.cpp, %.o, $(wildcard *.cpp)))

lenet_openmp: main_openmp.o lenet_openmp.o others.o mnist.o
	g++ -o lenet_openmp $(OUT_DIR)/main_openmp.o $(OUT_DIR)/lenet_openmp.o $(OUT_DIR)/others.o $(OUT_DIR)/mnist.o -lm -fopenmp 

lenet: main.o lenet.o others.o mnist.o
	g++ -o lenet $(OUT_DIR)/main.o $(OUT_DIR)/lenet.o $(OUT_DIR)/others.o $(OUT_DIR)/mnist.o -lm -fopenmp

main.o: main.cpp
	g++ -c main.cpp -o $(OUT_DIR)/$@ -fopenmp -DOPENMP=0

main_openmp.o: main.cpp
	g++ -c main.cpp -o $(OUT_DIR)/$@ -fopenmp -DOPENMP=1

lenet.o: lenet5/lenet.cpp
	g++ -c lenet5/lenet.cpp -o $(OUT_DIR)/$@ -DOPENMP=0

lenet_openmp.o: lenet5/lenet.cpp
	g++ -c lenet5/lenet.cpp -o $(OUT_DIR)/$@ -fopenmp -DOPENMP=1

others.o: lenet5/others.cpp
	g++ -c lenet5/others.cpp -o $(OUT_DIR)/$@

mnist.o: lenet5/mnist/mnist.cpp
	g++ -c lenet5/mnist/mnist.cpp -o $(OUT_DIR)/$@

# Aux
.PHONY: directories all clean

all: $(OUT_DIR)/program
all_openmp: $(OUT_DIR)/program_openmp

directories: $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)
	rm ./lenet

$(OUT_DIR):
	${MKDIR_P} $(OUT_DIR)

$(OUT_DIR)/program: | directories
	make lenet

$(OUT_DIR)/program_openmp: | directories
	make lenet_openmp
