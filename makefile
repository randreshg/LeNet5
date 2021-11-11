lenetcnn: main.o lenet.o backward.o forward.o others.o global.o mnist.o
	gcc -o lenetcnn main.o lenet.o backward.o forward.o others.o global.o mnist.o -lm

main.o: main.c
	gcc -c  main.c

lenet.o: lenet5/lenet.c
	gcc -c  lenet5/lenet.c

backward.o: lenet5/backward.c
	gcc -c  lenet5/backward.c

forward.o: lenet5/forward.c
	gcc -c  lenet5/forward.c

others.o: lenet5/others.c
	gcc -c  lenet5/others.c

global.o: lenet5/global/global.c
	gcc -c  lenet5/global/global.c

mnist.o: lenet5/mnist/mnist.c
	gcc -c  lenet5/mnist/mnist.c