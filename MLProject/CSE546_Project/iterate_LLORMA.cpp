#include "mex.h"
#include <math.h>

// Gradient descent for the LLORMA method for a single anchor point
// INPUTS: 
// Sparse matrix of ratings (M - of shape n_movies x n_users)
// Low rank decompositions U & V; U - shape (n_users x rank), V - (n_movies x rank)
// Weights for the Users and Movies K_user & K_mov; K_user - (n_users x 1), K_mov - (n_movies x 1)
// OUTPUTS:
// NONE- We operate on the input matrices U & V and modify them using gradient descent

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
    double  *pr, *pi;
    mwIndex  *ir, *jc;
    mwSize col,user,mov;
    mwIndex start_row, stop_row, curr_row;
    mwSize n_users,n_movies,rank;
    double *U, *V, *K_user, *K_mov;
    double Sum;
    
    // Parameters of the optimization
    double eta = 0.01; // Step size for gradient descent
    double lambda = 0.001; // Regularization parameter
    double epsilon = 0.0001; // Convergence criterion
    double maxIter = 100; // Maximum number of gradient descent iterations
    
    // Check if input matrix is sparse
    //if (!mxIsSparse(prhs[0]))
    //    mexErrMsgTxt(" Ratings matrix must be sparse ");

    /* Get the starting positions of the three data arrays for the sparse Matrix */ 
    pr = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    int rateCount = jc[mxGetN(prhs[0])];
    
    // Get the other matrices U, V, K_user, K_mov
    U = (double *)mxGetPr(prhs[1]);
    rank = mxGetN(prhs[1]); // Rank of the decomposition
    V = (double *)mxGetPr(prhs[2]);
    if (rank != mxGetN(prhs[2]))
        mexErrMsgTxt(" U and V must have the same number of columns ");
    
    // Get the weight matrices
    K_user = (double *)mxGetPr(prhs[3]); // User weights for current anchor point
    if (mxGetNumberOfElements(prhs[3]) != mxGetM(prhs[1]))
        mexErrMsgTxt(" We need as many weights as there are users ");
    K_mov = (double *)mxGetPr(prhs[4]); // Movie weights for current anchor point
    if (mxGetNumberOfElements(prhs[4]) != mxGetM(prhs[2]))
        mexErrMsgTxt(" We need as many weights as there are movies ");

    /* Display the nonzero elements of the sparse array. */ 
    n_users = mxGetN(prhs[0]); // Number of columns - n_users (input matrix is transposed)
    n_movies = mxGetM(prhs[0]); // no of rows is movies (since input is transposed)
    
    //printf(" Users: %d, Movies: %d \n", n_users, n_movies);
    // Now do gradient descent
    int round = 0;
    double prevErr = 99999;
    double currErr = 9999;
    double trainErr;
    while (fabs(prevErr - currErr) > epsilon && round < maxIter)
    {
        Sum = 0.0;

        for (col=0; col<n_users; col++)  // Iterate over users
        { 
            start_row = jc[col]; // Index to IR & PR to get first non zero rating for that user
            stop_row = jc[col+1]; // Index to IR & PR to get last non zero rating for that user
            user = col+1; // ID of the user

            if (start_row == stop_row) // meaning we have no data for that user
                continue;
            else 
            {
                for (curr_row = start_row; curr_row < stop_row; curr_row++)  //All movies for that user
                {
                    mov = ir[curr_row] + 1; // ID of the movie
                    double RuiEst = 0.0;
                    for (int r = 0; r < rank; r++) 
                    {
                        RuiEst += U[r*n_users + col] * V[r*n_movies + mov - 1];
                    }
                    double RuiReal = pr[curr_row];
                    //printf("U: %d, M: %d, Rating: %f \n",user,mov,pr[curr_row]);

                    double Err = RuiReal - RuiEst;
                    Sum += pow(Err,2);

                    double Weight = K_user[col] * K_mov[mov-1];

                    for (int r = 0; r < rank; r++) 
                    {
                        double Fus = U[r*n_users + col];
                        double Gis = V[r*n_movies + mov - 1];
                        U[r*n_users + col] = Fus + eta*(Err*Gis*Weight - lambda*Fus); // Note the minus due to M - UV' and not UV' - M
                        V[r*n_movies + mov - 1] = Gis + eta*(Err*Fus*Weight - lambda*Gis);
                    }
                }
            }
        }
        
        prevErr = currErr;
        currErr = Sum/rateCount;
        trainErr = sqrt(currErr);
			
        round++;
        printf("Loss function for iteration %d is : %f \n",round,trainErr);
    }
            
  
}