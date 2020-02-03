/* Author: TranDatDT
 * Release: 23:14 25/04/2018 GMT+7 */

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>
#define TEMPERATURE 4000
#define N 2048
#define THREAD_NUM 4

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
#   pragma omp parallel for
    for (int i = 0; i < size; i++) 
    {
        a_chessboard[i] = i + chess_board[i];
        m_chessboard[i] = i - chess_board[i];
    }

    std::sort(m_chessboard, m_chessboard + size);
    std::sort(a_chessboard, a_chessboard + size);

    int m_count = 1;
    int a_count = 1;
#   pragma omp parallel for reduction(+:m_count, a_count, threat)
    for (int i = 0; i < size - 1; i++) 
    {
        //int j = i + 1;
        if (m_chessboard[i] == m_chessboard[i+1])
        { 
            m_count += 1;
        }
        else 
        {
            threat += threat_calculate(m_count);
            m_count = 1;
        }
        if (a_chessboard[i] == a_chessboard[i+1]) 
        {
            a_count += 1;
        }
        else 
        {
            threat += threat_calculate(a_count);
            a_count = 1;
        }
        if (i + 1 == size - 1) 
        {
            threat += threat_calculate(m_count);
            threat += threat_calculate(a_count);
            //break;
        }
        
    }
    return threat;
}

int main() {
    clock_t start = clock();
    omp_set_num_threads(THREAD_NUM);

    srand((unsigned int) time(nullptr));
    std::random_device rd;
    std::mt19937 g(rd());
    unsigned int n_queens = N; // number of queens
    
    std::vector<int> final_answer;
 
    bool finish_flag=false;

#pragma omp parallel 
{
    int id;
    id = omp_get_thread_num();


    std::vector<int> answer;
    // std::cout << "Number of queens: ";
    // std::cin >> n_queens;

    // create a chess board
    answer.reserve(n_queens);

    for (int i = 0; i < n_queens; i++) 
    { // create a vector from 0 to N_QUEENS - 1
        answer.emplace_back(i);
    }

    std::shuffle(answer.begin(), answer.end(), g); //shuffle chess board to make sure it is random
    int cost_answer = cost(answer); // To avoid recounting in case can not find a better state

    // simulated annealing
    std::vector<int> successor;
    successor.reserve(n_queens);
    double t = TEMPERATURE;
    double sch = 0.99;
    while (!finish_flag) 
    {
        int rand_col_1;
        int rand_col_2;
        t *= sch;
        successor = answer;
        while (true) 
        { // random 2 queens
            rand_col_1 = (int) random() % n_queens;
            rand_col_2 = (int) random() % n_queens;
            if (successor[rand_col_1] != successor[rand_col_2]) 
                break;
        }
        std::swap(successor[rand_col_1], successor[rand_col_2]); // swap two queens chosen
        double delta = cost(successor) - cost_answer;
        if (delta < 0)
        {
            answer = successor;
            cost_answer = cost(answer);
        }
        else 
        {
            double p = exp(-delta / t);
            if (random() / double(RAND_MAX) < p) 
            {
                answer = successor;
                cost_answer = cost(answer);
            }
        }
        #pragma omp critical
        {
            if (cost_answer == 0) 
            {
                
                final_answer = answer;
                //print_chessboard(answer);
                printf("%d\n",id );
                finish_flag=true;
            }
        }

    }
}
    print_chessboard(final_answer);
    clock_t stop = clock();
    //std::cout << "Runtime: " << (float) (stop - start) / 1000000 << " seconds" << std::endl;

    return 0;
}
