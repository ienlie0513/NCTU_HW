# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <pthread.h>

long long int number_in_circle;
pthread_mutex_t mutex;

void* Thread_pi(void* number_of_tosses){ 
	unsigned int seed = time(NULL);
	long long int local_tosses = (long long int) number_of_tosses;
	long long int local_in_circle = 0;

	for(int toss=0; toss<local_tosses; toss++){
		double x = (double) rand_r(&seed) / RAND_MAX * 2.0 -1.0;
		double y = (double) rand_r(&seed) / RAND_MAX * 2.0 -1.0;

		double distance_squared = x * x + y * y ;
		if ( distance_squared <= 1) 
			local_in_circle ++;
	}
	pthread_mutex_lock(&mutex);
	number_in_circle += local_in_circle;
	pthread_mutex_unlock(&mutex);
}

int main(int argc, char *argv[]){
	//counter setting
	int cores;
	long long int number_of_tosses;

	cores = strtol(argv[1], NULL, 10);
	number_of_tosses = strtoll(argv[2], NULL, 10);

	//thread setting
	int thread_count = cores;
	long thread;
	pthread_t* thread_handles;
	thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t));
	pthread_mutex_init(&mutex, NULL);
	number_in_circle = 0;

	//create threads
	for (thread = 0; thread < thread_count; thread++)
		if (thread == 0 && number_of_tosses%thread_count != 0)
			pthread_create(&thread_handles[thread], NULL, Thread_pi, (void*)(number_of_tosses/thread_count+number_of_tosses%thread_count)); 
		else
			pthread_create(&thread_handles[thread], NULL, Thread_pi, (void*)(number_of_tosses/thread_count)); 

	//join threads
	for (thread = 0; thread < thread_count; thread++)
		pthread_join(thread_handles[thread], NULL);

	double pi_estimate = 4* number_in_circle /(( double ) number_of_tosses ) ;
	printf ("%f\n", pi_estimate);

	pthread_mutex_destroy(&mutex);
	free(thread_handles);
}
