#include<iostream>
#include<vector>
#include <chrono>
using namespace std;
#define N 64

int max_steps = 0;


//Function Desctiption: Calculates the total numner of conflicts for the Board
void CalculateConflictPairs(int board[], int& conflicts)
{
  conflicts = 0;
  for(int i = 0; i < N; i++)  //col
    for(int j = i+1; j < N; j++)  //row
      if((board[i] == board[j]) || (abs(i-j)==abs(board[i] - board[j])))
        conflicts++;
}

//Function Desctiption:  Calculates the number of conflicts for a particular
//         queen in the given column
void ConflictsCol(int board[], int &col, int &conflict)
{
  conflict = 0;
  for(int i = 0; i < col; i++)  // col
  {
    if(board[col] == board[i] || abs(col - i) == abs(board[col] - board[i]))
      conflict++;
  }
  for(int i = col+1; i < N; i++)    //col
  {
    if(board[col] == board[i] || abs(col - i) == abs(board[col] - board[i]))
      conflict++;
  }
}


/*
Function Desctiption: Solves N-queens problem using Random Restart Hill
  Climbing method.
Algorithm:
  1. Spwan N queens in random rows one in each column;
  2. Check for best possible successor(N*(N-1)) and select the best successor
  3. Swap the position of queen in the best successor row/column
  4. Calculate Total Conflicts
        if Conflicts = 0    then stop
        else if Conflicts <= parent Conflicts then restore swao and goto step 1.
        else goto step 2.
*/
void SolveUsingRandomRestartHillClimbing()
{
  int i,j, currentConflicts, tempConflicts, currentRow, bestSuccessorConflict;
  int bestSuccessorRow,bestSuccessorCol,numberOfRandomRestart=0, count = 0;
  int totalCount = 0;
  bool SolutionFound = false;
  int* QueensBoard = new int[N];
  auto begin = std::chrono::high_resolution_clock::now();
  while(!SolutionFound)         //loop for random restart
  {
    numberOfRandomRestart++;
    for(i = 0; i < N; i++)        //randomly populating the board
      QueensBoard[i] = rand() % N;
    CalculateConflictPairs(QueensBoard,currentConflicts);
    count= 0;
    if(currentConflicts == 0)
      SolutionFound = true;
    while(currentConflicts > 0)     //loop for hill climbing
    {
      totalCount++;
      count++;
      bestSuccessorConflict = currentConflicts;
      //checking all successor states i.e. M*(N-1) states
      for(i = 0; i < N; i++)  //col
      {
        currentRow = QueensBoard[i];
        for(j = 0; j < N; j++)    //row
        {
          if(currentRow != j)
          {
              QueensBoard[i] = j;
              CalculateConflictPairs(QueensBoard,tempConflicts);
              //selecting the best successor
              if(tempConflicts < bestSuccessorConflict)
              {
                bestSuccessorConflict = tempConflicts;
                bestSuccessorRow = j;
                bestSuccessorCol = i;
              }
          }
        }
        QueensBoard[i] = currentRow;
      }
      if(bestSuccessorConflict >= currentConflicts) //local maxima or shoulder
        break;
      else                                  //swapping with the best successor
      {
        QueensBoard[bestSuccessorCol] = bestSuccessorRow;
        currentConflicts = bestSuccessorConflict;
      }
      if(bestSuccessorConflict == 0)  //solution found
      {
        SolutionFound = true;
        break;
      }
    }//inner while
  }//outer while
  auto end = std::chrono::high_resolution_clock::now();
  cout<<"Solution Found ::"<<endl;
  for(i = 0; i < N; i++)    //printing Board
  {
      for(j = 0; j < N; j++)
      {
          if(QueensBoard[i] == j)
            cout<<"Q|";
          else
            cout<<"_|";
      }
      cout<<endl;
  }
  cout<<endl;
  cout << "Time taken to solve :: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1000000000.0 << "s";
  cout<<"\nNumber of Random Restart  "<<numberOfRandomRestart<<endl;
  cout<<"Total Number of states generated :: "<<totalCount<<endl;
  cout<<"Total States generated from the most random restart :: "<<count<<endl;
  delete[] QueensBoard;
}


/*
Function Desctiption: Solves the M-Queens problem using Min Conflicts method as
  a CSP
Algorithm::
  1. Spwan N queens in random rows one in each column;
  2. Select a random column which has cnflicts and find the numnber of
     conflicting pair for that column
  3. Select a vakue for the column that gives lesser number of conflicts than
     current value
  4. if numnber of steps exceeds max_count then goto step 1
     else if Total conflicts = 0 then return Solution
     else gptp step 3.
*/
void SolveUsingMinConflict()
{
  int i,j, currentConflicts =-1, currentRow, tempConflicts, bestSuccessorRow;
  int bestSuccessorConflict,numberOfRandomRestart=0, totalCount = 0;
  int randomCol,count, colConflicts, tempcolConflicts, lastRandomCol;
  bool SolutionFound = false;
  int* QueensBoard = new int[N];
  auto begin = std::chrono::high_resolution_clock::now();
  while(!SolutionFound)       //loop for random restart
  {
    if(currentConflicts == 0)
    {
        SolutionFound = true;
        break;
    }
    count = 0;
    numberOfRandomRestart++;
    for(i = 0; i < N; i++)        //spwaning queens randomly
      QueensBoard[i] = rand() % N;
    CalculateConflictPairs(QueensBoard,currentConflicts);
    while(currentConflicts > 0 && count < max_steps)    //loop for max_count
    {
      count++;
      totalCount++;
      vector<int> cols_with_collisions;
      for (i = 0; i < N; i++)           //selecting columns with conflicts
      {
        ConflictsCol(QueensBoard, i, colConflicts);
        if (colConflicts > 0)
          cols_with_collisions.push_back(i);
      }
      //selecting a random column with conflict
      randomCol = cols_with_collisions[rand()%cols_with_collisions.size()];

      currentRow = QueensBoard[randomCol];
      ConflictsCol(QueensBoard, randomCol, colConflicts);
      vector<int> successors;
      for(i = 0; i < N; i++)      //looking for a better value of conflict
      {
        QueensBoard[randomCol] = i;
        ConflictsCol(QueensBoard, randomCol, tempConflicts);
        if(tempConflicts < colConflicts)
          successors.push_back(i);
      }
      // minimizing the number of conflicts if possible
      if(successors.size() > 0)
        QueensBoard[randomCol] = successors[rand()%successors.size()];
      else
        QueensBoard[randomCol] = currentRow;
      CalculateConflictPairs(QueensBoard,currentConflicts);
    }//inner while loop
  }//outer while loop
  auto end = std::chrono::high_resolution_clock::now();
  cout<<"Solution Found ::"<<endl;
  for(i = 0; i < N; i++)    //printing Board
  {
      for(j = 0; j < N; j++)
      {
          if(QueensBoard[i] == j)
            cout<<"Q|";
          else
            cout<<"_|";
      }
      cout<<endl;
  }
  cout<<endl;
  cout << "Time taken to solve :: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1000000000.0 << "s";
  cout<<"\nNumber of Random Restart  "<<numberOfRandomRestart<<endl;
  cout<<"Total Number of states generated :: "<<totalCount<<endl;
  cout<<"Total States generated from the most random restart :: "<<count<<endl;
  delete[] QueensBoard;
}

int main(void)
{
  //taking input from the user
  // while(N<4)
  // {
  //   cout<<"Enter N (N>=4) :: ";
  //   cin>>N;
  // }

  int choice = 0;
  srand(time(NULL));
  //select the Algorithm
  SolveUsingRandomRestartHillClimbing();
  // while(choice < 1 || choice > 2)
  // {
  //   cout<<"Select one of the following  \n\t1. Random Restart Hill Climbing\n\t2. Min Conflict N-Queen\nEnter your choice :: ";
  //   cin>>choice;
  //   switch (choice) {
  //     case 1:
  //       SolveUsingRandomRestartHillClimbing();
  //     break;
  //     case 2:
  //     {
  //       cout<<"Enter Max steps (Preferred value is same as N):: ";
  //       cin>>max_steps;
  //       max_steps += N;
  //       SolveUsingMinConflict();
  //     }
  //     break;
  //     default: cout<<"Incorrect input\n";
  //   }
  // }
  return 0;
}