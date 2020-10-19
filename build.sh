gcc -O3 main.cpp ./src/clut.cpp -Iinclude/ -lm -lstdc++ `pkg-config --cflags --libs opencv` -o main
