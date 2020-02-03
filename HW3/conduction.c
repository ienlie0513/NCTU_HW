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
	
	int my_rank; // my process rank
	int p; // number of processes

	/* set MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* init command line parameters */
	if (my_rank==0) {
		L = atoi(argv[1]); // Length
		iteration = atoi(argv[2]); // Iteration
		srand(atoi(argv[3])); // Seed

		d = (float) random() / RAND_MAX * 0.2; // Diffusivity
	}

	/* broadcast parameters */
	MPI_Bcast(&L, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&iteration, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&d, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	int *temp = malloc(L*W*sizeof(int)); // Current temperature

	/* init temp */
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

	
	/*prepare each worker*/
	int local_size = L/p+2;
	int *local_temp = malloc(local_size*W*sizeof(int));
	int *local_next = malloc(local_size*W*sizeof(int)); // Next time step
	int li = 0;
	int begin_row = (my_rank == 0 ? 0 : L/p*my_rank-1);
	int end_row = (my_rank == p-1 ? L : L/p*(my_rank+1)+1);
	
	// copy the top array of rank==0
	if (my_rank == 0){
		for (int j = 0; j < W; j++){
			local_temp[li] = temp[j];
			li++;
		}
	}
	// copy the middle part of array
	for (int i = begin_row; i < end_row; i++){
		for (int j = 0; j < W; j++){
			local_temp[li] = temp[i*W+j];
			li++;
		}
	}
	// copy the button array of rank==p-1
	if (my_rank == p-1){
		for (int j = 0; j < W; j++){
			local_temp[li] = temp[(L-1)*W+j];
			li++;
		}
	}


	int count = 0, balance = 0;
	while (iteration--) { // Compute with up, left, right, down points
		count++;
		balance = 1;

		for (int i = 1; i < local_size-1; i++) {
			for (int j = 0; j < W; j++) {
				float t = local_temp[i*W+j] / d;
				t += local_temp[i*W+j] * -4;
				t += local_temp[(i - 1) * W + j];
				t += local_temp[(i + 1)*W+j];
				t += local_temp[i*W+(j - 1 < 0 ? 0 : j - 1)];
				t += local_temp[i*W+(j + 1 >= W ? j : j + 1)];
				t *= d;
				local_next[i*W+j] = t;
				if (local_next[i*W+j] != local_temp[i*W+j]) {
					balance = 0;
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);		
		//check if balance
		MPI_Allreduce(&balance, &balance, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (balance == L*W) {
			break;
		}

		// data transform
		int tag = 0;
		MPI_Status status; 
		// send from local_next and recv to local_next
		if (my_rank == 0) {
			// transform with rank == 1 
			MPI_Send(&local_next[(local_size-2)*W], W, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD);
			MPI_Recv(&local_next[(local_size-1)*W], W, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD, &status);
			// copy self
			for (int j = 0; j < W; j++){
				local_next[j] = local_next[1*W+j];
			}
		}
		else if (my_rank == p-1) {
			// transform with rank == p-1
			MPI_Recv(&local_next[0], W, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, &status);
			MPI_Send(&local_next[1*W], W, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD);
			// copy self
			for (int j = 0; j < W; j++){
				local_next[(local_size-1)*W+j] = local_next[(local_size-2)*W+j];
			}
		}
		else {
			// transform with last rank 
			MPI_Recv(&local_next[0], W, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD, &status);
			MPI_Send(&local_next[1*W], W, MPI_INT, my_rank-1, tag, MPI_COMM_WORLD);
			// transform with next rank  
			MPI_Send(&local_next[(local_size-2)*W], W, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD);
			MPI_Recv(&local_next[(local_size-1)*W], W, MPI_INT, my_rank+1, tag, MPI_COMM_WORLD, &status);
		}

		int *tmp = local_temp;
		local_temp = local_next;
		local_next = tmp;
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	// gather data from each worker
	MPI_Gather(&local_temp[1*W], L/p*W, MPI_INT, temp, L/p*W, MPI_INT, 0, MPI_COMM_WORLD);
	

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
	MPI_Finalize();

	return 0;
}
