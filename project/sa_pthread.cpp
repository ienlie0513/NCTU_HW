/* Author: TranDatDT
 * Release: 23:14 25/04/2018 GMT+7 */

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <pthread.h>
#define TEMPERATURE 4000
#define N 2048

std::vector<int> final_answer;
long flag;
pthread_mutex_t mutex;

void print_chessboard(std::vector<int> chess_board) { // print the chessboard
    for (int queen = 0; queen < chess_board.size(); queen++)
        std::cout << queen << " => " << chess_board[queen] << "\n";
}

int threat_calculate(int n) { // combination formula for calculate number of pairs of threaten queens
    if (n < 2) return 0;
    if (n == 2) return 1;
    return (n - 1) * n / 2;
}

int cost(std::vector<int> chess_board) { // cost function to count total of pairs of threaten queens
    unsigned long size = chess_board.size();
    int threat = 0;
    int m_chessboard[size];
    int a_chessboard[size];

    for (int i = 0; i < size; i++) {
        a_chessboard[i] = i + chess_board[i];
        m_chessboard[i] = i - chess_board[i];
    }

    std::sort(m_chessboard, m_chessboard + size);
    std::sort(a_chessboard, a_chessboard + size);

    // for (int i = 0; i < size; i++)
    //     std::cout << chess_board[i] << " ";
    // std::cout << std::endl;
    // for (int i = 0; i < size; i++)
    //     std::cout << m_chessboard[i] << " ";
    // std::cout << std::endl;
    // for (int i = 0; i < size; i++)
    //     std::cout << a_chessboard[i] << " ";
    // std::cout << std::endl;

    int m_count = 1;
    int a_count = 1;

    for (int i = 0; i < size - 1; i++) {
        int j = i + 1;
        if (m_chessboard[i] == m_chessboard[j]) m_count += 1;
        else {
            threat += threat_calculate(m_count);
            m_count = 1;
        }
        if (a_chessboard[i] == a_chessboard[j]) a_count += 1;
        else {
            threat += threat_calculate(a_count);
            a_count = 1;
        }
    }
    threat += threat_calculate(m_count);
    threat += threat_calculate(a_count);

    return threat;
}

void* simulated_annealing(void* rank){
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> answer;
    unsigned int n_queens = N;

    long myrank = (long)rank;

    // create a chess board
    answer.reserve(n_queens);
    for (int i = 0; i < n_queens; i++) { // create a vector from 0 to N_QUEENS - 1
        answer.emplace_back(i);
    }
    std::shuffle(answer.begin(), answer.end(), g); //shuffle chess board to make sure it is random
    int cost_answer = cost(answer); // To avoid recounting in case can not find a better state
    // std::cout << cost_answer << std::endl;

    // simulated annealing
    std::vector<int> successor;
    successor.reserve(n_queens);
    double t = TEMPERATURE;
    double sch = 0.99;
    while (t > 0) {
        int rand_col_1;
        int rand_col_2;
        t *= sch;
        successor = answer;
        while (true) { // random 2 queens
            rand_col_1 = (int) random() % n_queens;
            rand_col_2 = (int) random() % n_queens;
            if (successor[rand_col_1] != successor[rand_col_2]) break;
        }
        std::swap(successor[rand_col_1], successor[rand_col_2]); // swap two queens chosen
        double delta = cost(successor) - cost_answer;
        if (delta < 0){
            answer = successor;
            cost_answer = cost(answer);
        }
        else {
            double p = exp(-delta / t);
            if (random() / double(RAND_MAX) < p) {
                answer = successor;
                cost_answer = cost(answer);
            }
        }
        if (cost_answer == 0) { //find answer
            pthread_mutex_lock(&mutex);
            final_answer = answer;
            flag = myrank;
            pthread_mutex_unlock(&mutex);
            return NULL;
        }
    }
}

int main(int argc, char *argv[]) {
    clock_t start = clock();
    srand((unsigned int) time(nullptr));

    flag = -1;

    // std::cout << "Number of queens: ";
    // std::cin >> n_queens;

    //thread setting
    int thread_count = strtol(argv[1], NULL, 10);
    long thread;
    pthread_t* thread_handles;
    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t));

    //create threads
    for (thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, simulated_annealing, (void*)thread); 

    while(flag == -1){}
    
    for (thread = 0; thread < thread_count; thread++){
        if(thread != flag){
            int err = pthread_cancel(thread_handles[thread]);
            if (err != 0) 
                printf("Thread cancel error %d\n", err);
        }
    }
    
    print_chessboard(final_answer);    

    clock_t stop = clock();
    std::cout << "Runtime: " << (float) (stop - start) / 1000000 << " seconds" << std::endl;

    return 0;
}