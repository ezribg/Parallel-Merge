#include <iostream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <iomanip>
#include <fstream>
#include "mpi.h" // message passing interface
//using namespace std;

// New compile and run commands for MPI!
// mpicxx -o blah file.cpp
// mpirun -q -np 32 blah   //32 = processer number  //currently can only use 24

void smerge(int * a, int * b, int lasta, int lastb, int * output);
void mergesort (int * a, int first, int last, int * og);
int rank(int * a, int first, int last, int valToFind);
void pmerge(int * a, int * b, int lasta, int lastb, int * output);

int my_rank;			// my CPU number for this process
int p;					// number of CPUs that we have

int main (int argc, char * argv[]) {
    
    int source;				// rank of the sender
	int dest;				// rank of destination
	int tag = 0;			// message number
    MPI_Status status;		// return status for receive

    // Start MPI
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    //process 0 - create inital array a
    int n = 64;
    int seed = 28934;
    srand(seed);
    int * array = new int[n];

    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            array[i] = rand() % 200;
            std::cout << array[i] << "  ";
        }
        std::cout << std::endl;
        
    }

    MPI_Bcast(&array[0], n, MPI_INT, 0, MPI_COMM_WORLD);

    mergesort(array, 0, n, array);

    if (my_rank == 0) {
        for (int i = 0; i < n; i++)
            std::cout << array[i] << "  ";
        std::cout << std::endl;
    }

	// Shut down MPI
	MPI_Finalize();

	return 0;
}



void mergesort (int * a, int first, int last, int * og) {

    if (last < 2)
        return;
    
    if (last == 2) {
        if (a[0] > a[1]) {
            int temp = a[0];
            a[0] = a[1];
            a[1] = temp;
        }
        return;
    }

    int q = last / 2;
    int * right = &a[q];

    mergesort(a, first, q, og);
    mergesort(right, first, last - q, og);

    int * tempArr = new int[last];

    for (int i = 0; i < last; i++)
        tempArr[i] = 0;

    pmerge(a, right, q, last - q, tempArr);

    for (int i = 0; i < last; i++) {
        a[i] = tempArr[i];
    }

    delete[] tempArr;
}


int rank(int * a, int first, int last, int valToFind) {
    if (last == 1) {
        if (valToFind <= a[first]) {
            return 0;
        }
        else { return 1; }
    }
    else {
        int x = last / 2;
        int * right = &a[x];
        if (valToFind < a[x]) {
            return rank(a, first, x, valToFind);
        }
        else {
            return x + rank(right, first, last - x, valToFind);
        }
    }
}

void smerge(int * a, int * b, int lasta, int lastb, int * output = NULL) {
    int i = 0, j = 0, k = 0;

    while (i < lasta && j < lastb) {
        if (a[i] < b[j])
            output[k++] = a[i++];
        else
            output[k++] = b[j++];
    }
    
    for (int z = i; z < lasta; z++)
        output[k++] = a[z];
    
    for (int z = j; z < lastb; z++)
        output[k++] = b[z];
    
}

void pmerge(int * a, int * b, int lasta, int lastb, int * output = NULL) {

    // Stage 1
    int x = floor(lasta/log2(lasta));
    int y = floor(lastb/log2(lastb));
    int * totsranka = new int[x];
    int * totsrankb = new int[y];
    int * pivots = new int[x];
    
    // Fill in select and srank lists
    int a_select[x];
    int * sranka = new int[x];
    for (int i = 0; i < x; i++) {
        a_select[i] = a[i * x];
        sranka[i] = 0;
        totsranka[i] = 0;
        pivots[i] = i * x;
    }

    for (int i = my_rank; i < x; i += p)
        sranka[i] = rank(b, 0, lastb, a_select[i]);

    int b_select[y];
    int * srankb = new int[y];
    for (int i = 0; i < y; i++) {
        b_select[i] = b[i * y];
        srankb[i] = 0;
        totsrankb[i] = 0;
    }

    for (int i = my_rank; i < y; i += p)
        srankb[i] = rank(a, 0, lasta, b_select[i]);

    // Bring together
    MPI_Allreduce(&sranka[0], &totsranka[0], x, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&srankb[0], &totsrankb[0], y, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Find shape endpoints
    int * endpointA = new int[x * 2 + 1];
    int * endpointB = new int[y * 2 + 1];
    int * tempwin = new int[lasta + lastb];

    for (int i = 0; i < lasta + lastb; i++)
        tempwin[i] = 0;

    smerge(pivots, totsrankb, x, y, endpointA);
    smerge(pivots, totsranka, x, y, endpointB);

    endpointA[x * 2] = lasta;
    endpointB[y * 2] = lastb;
    
    for (int i = my_rank; i < x * 2; i += p) {
        smerge(&a[endpointA[i]], &b[endpointB[i]], endpointA[i + 1] - endpointA[i], endpointB[i + 1] - endpointB[i], &tempwin[endpointA[i] + endpointB[i]]);
    }

    /*std::cout << "tempwin " << my_rank << " ";
    for (int x = 0; x < lasta + lastb; x++)
        std::cout << tempwin[x] << " ";
    std::cout << std::endl;*/

    MPI_Allreduce(&tempwin[0], &output[0], lasta + lastb, MPI_INT, MPI_MAX, MPI_COMM_WORLD);


    delete[] totsranka;
    delete[] totsrankb;
    delete[] sranka;
    delete[] srankb;
    delete[] endpointA;
    delete[] endpointB;
    delete[] pivots;
}