#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#ifndef W
#define W 20 // Width
#endif

int main(int argc, char **argv) {
	int L; // Length
	int iteration; // Iteration
	float d; // Diffusivity
	
	/* my process rank */
	int my_rank;
	/* The number of processes */
	int p;

	/* init MPI */
	MPI_Init(&argc, &argv);
	/* Get my process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Find out how many processes are being used */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* init command line parameters */
	if (my_rank==0) {
		L = atoi(argv[1]); // Length
		iteration = atoi(argv[2]); // Iteration
		srand(atoi(argv[3])); // Seed

		d = (float) random() / RAND_MAX * 0.2; // Diffusivity
	}
	/* broadcast */
	MPI_Bcast(&L, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&iteration, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&d, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	int *temp = malloc(L*W*sizeof(int)); // Current temperature
	int *next = malloc(L*W*sizeof(int)); // Next time step

	/* init parameters */
	if (my_rank==0) {
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < W; j++) {
				temp[i*W+j] = random()>>3;
			}
		}
	}

	/* broadcast temp */
	MPI_Bcast(temp, L*W, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	int count = 0, balance = 0;
	while (iteration--) { // Compute with up, left, right, down points
		count++;
		balance = 1;
		//local next and i for each workers 
		int *local_next = malloc(L/p*W*sizeof(int));
		int l_i = 0;
		for (int i = L/p*my_rank; i < L/p*(my_rank+1); i++) {
			for (int j = 0; j < W; j++) {
				float t = temp[i*W+j] / d;
				t += temp[i*W+j] * -4;
				t += temp[(i - 1 < 0 ? 0 : i - 1) * W + j];
				t += temp[(i + 1 >= L ? i : i + 1)*W+j];
				t += temp[i*W+(j - 1 < 0 ? 0 : j - 1)];
				t += temp[i*W+(j + 1 >= W ? j : j + 1)];
				t *= d;
				local_next[l_i*W+j] = t;
				// next[i*W+j] = t;
				if (local_next[l_i*W+j] != temp[i*W+j]) {
					balance = 0;
				}
			}
			l_i++;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		// gather next and reduce balance 
		MPI_Allgather(local_next, L/p*W, MPI_INT, temp, L/p*W, MPI_INT, MPI_COMM_WORLD);
		free(local_next);
		//check if balance
		MPI_Allreduce(&balance, &balance, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (balance == L*W) {
			break;
		}
	}
	
	if (my_rank==0) {
		int min = temp[0];
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < W; j++) {
				if (temp[i*W+j] < min) {
					min = temp[i*W+j];
				}
			}
		}

		printf("Size: %d*%d, Iteration: %d, Min Temp: %d\n", L, W, count, min);
	}


	free(temp);
	free(next);
	MPI_Finalize();

	return 0;
}
