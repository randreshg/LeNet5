lenetcnn: main.o lenet.o backward.o forward.o others.o global.o
	gcc -o lenetcnn main.o lenet.o backward.o forward.o others.o global.o -lm

main.o: main.c
	gcc -c -g main.c

lenet.o: lenet5/lenet.c
	gcc -c -g lenet5/lenet.c

backward.o: lenet5/backward.c
	gcc -c -g lenet5/backward.c

forward.o: lenet5/forward.c
	gcc -c -g lenet5/forward.c

others.o: lenet5/others.c
	gcc -c -g lenet5/others.c

global.o: lenet5/global/global.c
	gcc -c -g lenet5/global/global.c