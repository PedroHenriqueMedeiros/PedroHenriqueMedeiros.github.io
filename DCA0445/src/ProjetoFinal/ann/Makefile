.SUFFIXES:
.SUFFIXES: .c .cpp

CC = gcc
GCC = g++

.c:
	$(CC) -I$(INCDIR) $(CFLAGS) $< $(GL_LIBS) -o $@

.cpp:
	$(GCC) -Wall -Wunused -g -std=c++11 -O2 `pkg-config --cflags opencv` $< -o $@ `pkg-config --libs opencv` -lopencv_nonfree -lboost_filesystem -lboost_system -lboost_thread


