#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>


/** @name Linear algebra wrappers */
///@{

/**
 \brief Multiply matrices \f$ AB = A\timesB\f$

 @param[in] A \f$N \times N\f$ matrix
 @param[in] B \f$N \times N\f$ matrix
 @param[in] N size of matrices
 @param[out] AB \f$A\timesB\f$
 */
void matrix_multiply(double **A, double **B, double **AB, int N);

/**
 \brief Wrapper to GSL matrix inversion functions.

 @param[in,out] matrix input matrix to be inverted. Inverse is stored in-place. Input matrix is lost.
 @param[in] N size of input matrix (\f$ N\timesN\f$)
 @param[out] det determinent of input matrix
 */
void invert_matrix(double **matrix, double *det, int N);

///@}

/**
 \brief Likelihood function
 
 Multivariate Gaussian log likelihood (assumes 0 mean) \f$ \log L = -\frac{1}{2}(x^T C^{-1} x + \det C) \f$
 
 @param[in] x vector of parameter values
 @param[in] C covariance matrix
 @param[in] N size of vector
 @return \f$\log L\f$
*/
double loglikelihood(double *x, double **C, int N);

/** @name Gibbs/MCMC updates */
///@{

/**
 \brief Gibbs update of `x` parameter
 
 @param[in] x vector of parameter values
 @param[in] C covariance matrix
 @param[in] r random number state
 @param[in] N size of vector
 @param[out] x[0] updated `x` parameter
 */
void update_x(double *x, double **C, gsl_rng *r, int N);

/**
 \brief Gibbs update of `y` parameter
 
 @param[in] x vector of parameter values
 @param[in] C covariance matrix
 @param[in] r random number state
 @param[in] N size of vector
 @param[out] x[1] updated `y` parameter
 */
void update_y(double *x, double **C, gsl_rng *r, int N);

/**
 \brief Gibbs update of `z` parameter
 
 @param[in] x vector of parameter values
 @param[in] C covariance matrix
 @param[in] r random number state
 @param[in] N size of vector
 @param[out] x[2] updated `z` parameter
 */
void update_z(double *x, double **C, gsl_rng *r, int N);

/**
 \brief MCMC update of `x,y,z` parameters
 
 @param[in] x vector of parameter values
 @param[in] C covariance matrix
 @param[in] r random number state
 @param[in] N size of vector
 @param[out] x updated parameter vector
 */
void update_xyz(double *x, double **C, gsl_rng *r, int N);
///@}

int main(int argc, char *argv[])
{
    int NMCMC = 500000; //number of MCMC steps
    int Nproc, procID;  //keep track of which MPI process

    MPI_Init(&argc, &argv); //start parallelization

    /* get process ID, and total number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &Nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);
    
    /* set up covariance matrix */
    double **C = malloc(Nproc*sizeof(double *));
    for(int n=0; n<Nproc; n++) C[n] = malloc(Nproc*sizeof(double));
    
    //make correlation matrix
    double **corr = malloc(Nproc*sizeof(double *));
    for(int n=0; n<Nproc; n++) corr[n] = malloc(Nproc*sizeof(double));
    corr[0][0]=1.0;
    corr[1][1]=1.0;
    corr[2][2]=1.0;
    /* big biased
    corr[0][1] = corr[1][0] = -0.6;
    corr[0][2] = corr[2][0] = 0.95;
    corr[2][1] = corr[1][2] = -0.8;
     */
    /* medium bias
    corr[0][1] = corr[1][0] = -0.6;
    corr[0][2] = corr[2][0] = 0.90;
    corr[2][1] = corr[1][2] = -0.8;
     */
    /* small bias
    corr[0][1] = corr[1][0] = -0.5;
    corr[0][2] = corr[2][0] = 0.8;
    corr[2][1] = corr[1][2] = -0.7;
     */
    corr[0][1] = corr[1][0] = -0.1;
    corr[0][2] = corr[2][0] = 0.5;
    corr[2][1] = corr[1][2] = -0.3;

    
    //make variance matrix
    double **var = malloc(Nproc*sizeof(double *));
    for(int n=0; n<Nproc; n++) var[n] = malloc(Nproc*sizeof(double));
    var[0][0] = 0.1;
    var[1][1] = 0.01;
    var[2][2] = 1.0;
    var[0][1] = var[1][0] = 0.0;
    var[0][2] = var[2][0] = 0.0;
    var[2][1] = var[1][2] = 0.0;

    //make covariance var * corr * var
    double **temp = malloc(Nproc*sizeof(double *));
    for(int n=0; n<Nproc; n++) temp[n] = malloc(Nproc*sizeof(double));
    
    matrix_multiply(corr, var, temp, Nproc);
    matrix_multiply(var, temp, C, Nproc);

    /* initialize model parameters */
    double *x = malloc(Nproc*sizeof(double));
    for(int n=0; n<Nproc; n++) x[n]=0.0;
    
    /* set up RNGs */
    long seed = (long)procID;
    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng *r = gsl_rng_alloc(T);
    gsl_rng_env_setup();
    gsl_rng_set(r,seed);
    
    char filename[128];
    FILE *chainFile;
    
    /* standard MCMC sampler */
    if(procID==0)printf("Standard sampler...\n");
    sprintf(filename,"standard_mcmc_%i.dat",procID);
    chainFile = fopen(filename,"w");
    
    for(int n=0; n<NMCMC; n++)
    {
        //joint update of full parameter set
        for(int i=0; i<1; i++)update_xyz(x, C, r, Nproc);
        for(int i=0; i<3; i++)update_xyz(x, C, r, Nproc);
        for(int i=0; i<10; i++)update_xyz(x, C, r, Nproc);

        for(int i=0; i<Nproc; i++) fprintf(chainFile,"%lg ",x[i]);
        fprintf(chainFile,"\n");
    }
    
    fclose(chainFile);/**/

    
    /* parallel gibbs sampler */
    if(procID==0)printf("Parallel Gibbs sampler...\n");
    if(procID==0)
    {
        sprintf(filename,"parallel_gibbs.dat");
        chainFile = fopen(filename,"w");
    }
    for(int n=0; n<NMCMC; n++)
    {
        // each process executes a number of Gibbs updates
        if(procID==0) for(int i=0; i<1; i++) update_x(x, C, r, Nproc);
        if(procID==1) for(int i=0; i<3; i++) update_y(x, C, r, Nproc);
        if(procID==2) for(int i=0; i<10; i++) update_z(x, C, r, Nproc);

        // all processes exchange information
        MPI_Bcast(x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(x+1, 1, MPI_DOUBLE, 1, MPI_COMM_WORLD);
        MPI_Bcast(x+2, 1, MPI_DOUBLE, 2, MPI_COMM_WORLD);

        if(procID==0)
        {
            for(int i=0; i<Nproc; i++) fprintf(chainFile,"%lg ",x[i]);
            fprintf(chainFile,"\n");
        }
    }
    
    if(procID==0)fclose(chainFile);

    
    /* series gibbs sampler */
    if(procID==0)printf("Serial Gibbs sampler...\n");
    if(procID==0)
    {
        sprintf(filename,"serial_gibbs.dat");
        chainFile = fopen(filename,"w");
    }
    for(int n=0; n<NMCMC; n++)
    {
        // one process at a time executes a number of Gibbs updates.  All others wait.
        if(procID==0) for(int i=0; i<1; i++) update_x(x, C, r, Nproc);
        MPI_Bcast(x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(procID==1) for(int i=0; i<3; i++) update_y(x, C, r, Nproc);
        MPI_Bcast(x+1, 1, MPI_DOUBLE, 1, MPI_COMM_WORLD);

        if(procID==2) for(int i=0; i<10; i++) update_z(x, C, r, Nproc);
        MPI_Bcast(x+2, 1, MPI_DOUBLE, 2, MPI_COMM_WORLD);

        if(procID==0)
        {
            for(int i=0; i<Nproc; i++) fprintf(chainFile,"%lg ",x[i]);
            fprintf(chainFile,"\n");
        }
    }
    
    if(procID==0)fclose(chainFile);/**/

    
    MPI_Finalize();//ends the parallelization

    return 0;
}


void matrix_multiply(double **A, double **B, double **AB, int N)
{
    //AB = A*B
    
    int i,j,k;
    
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            AB[i][j] = 0.0;
            for(k=0; k<N; k++)
            {
                AB[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    
}

void invert_matrix(double **matrix, double *det, int N)
{
    int i,j;
    
    // Don't let errors kill the program (yikes)
    gsl_set_error_handler_off ();
    int err=0;
    
    // Find eigenvectors and eigenvalues
    gsl_matrix *GSLmatrix = gsl_matrix_alloc(N,N);
    gsl_matrix *GSLinvrse = gsl_matrix_alloc(N,N);
    
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            if(matrix[i][j]!=matrix[i][j])fprintf(stderr,"nan matrix element at line %d in file %s\n", __LINE__, __FILE__);
            gsl_matrix_set(GSLmatrix,i,j,matrix[i][j]);
        }
    }
    
    gsl_permutation * permutation = gsl_permutation_alloc(N);
    
    err += gsl_linalg_LU_decomp(GSLmatrix, permutation, &i);
    err += gsl_linalg_LU_invert(GSLmatrix, permutation, GSLinvrse);
    
    *det = gsl_linalg_LU_det(GSLmatrix, i);
    
    if(err>0)
    {
        fprintf(stderr,"GalacticBinaryMath.c:184: WARNING: singluar matrix\n");
        fflush(stderr);
    }
    else
    {
        //copy inverse back into matrix
        for(i=0; i<N; i++)
        {
            for(j=0; j<N; j++)
            {
                matrix[i][j] = gsl_matrix_get(GSLinvrse,i,j);
            }
        }
    }
    
    gsl_matrix_free (GSLmatrix);
    gsl_matrix_free (GSLinvrse);
    gsl_permutation_free (permutation);
}

double loglikelihood(double *x, double **C, int N)
{
    double detC;
    double **invC = malloc(N*sizeof(double *));
    for(int n=0; n<N; n++)
    {
        invC[n] = malloc(N*sizeof(double));
        for(int m=0; m<N; m++)
        {
            invC[n][m] = C[n][m];
        }
    }
    
    invert_matrix(invC,&detC,N);
    
    if(detC<0.0)
    {
        printf("singular matrix, try again: det=%g\n",detC);
        exit(1);
    }
    
    double chi2 = 0.0;
    for(int n=0; n<N; n++)
    {
        for(int m=0; m<N; m++)
        {
            chi2 += x[n]*x[m]*invC[n][m];
        }
    }
    
    double logL = -0.5*log(detC) - 0.5*chi2;
    
    
    for(int n=0; n<2; n++) free(invC[n]);
    free(invC);
    
    return logL;
}

void update_x(double *x, double **C, gsl_rng *r, int N)
{
    double *xtemp = malloc(N*sizeof(double));
    for(int n=0; n<N; n++) xtemp[n] = x[n];
    
    xtemp[0] = x[0] + 0.1*gsl_ran_gaussian(r,1);
    
    double logH = loglikelihood(xtemp,C,N) - loglikelihood(x,C,N);
    
    double alpha = log(gsl_rng_uniform(r));
    
    if(logH>alpha) x[0] = xtemp[0];
    
    free(xtemp);
}

void update_y(double *x, double **C, gsl_rng *r, int N)
{
    double *xtemp = malloc(N*sizeof(double));
    for(int n=0; n<N; n++) xtemp[n] = x[n];
    
    xtemp[1] = x[1] + 0.01*gsl_ran_gaussian(r,1);
    
    double logH = loglikelihood(xtemp,C,N) - loglikelihood(x,C,N);
    
    double alpha = log(gsl_rng_uniform(r));
    
    if(logH>alpha) x[1] = xtemp[1];
    
    free(xtemp);
}

void update_z(double *x, double **C, gsl_rng *r, int N)
{
    double *xtemp = malloc(N*sizeof(double));
    for(int n=0; n<N; n++) xtemp[n] = x[n];
    
    xtemp[2] = x[2] + 1.0*gsl_ran_gaussian(r,1);
    
    double logH = loglikelihood(xtemp,C,N) - loglikelihood(x,C,N);
    
    double alpha = log(gsl_rng_uniform(r));
    
    if(logH>alpha) x[2] = xtemp[2];
    
    free(xtemp);
}

void update_xyz(double *x, double **C, gsl_rng *r, int N)
{
    double *xtemp = malloc(N*sizeof(double));
    for(int n=0; n<N; n++) xtemp[n] = x[n];
    
    xtemp[0] = x[0] + 0.1*gsl_ran_gaussian(r,1);
    xtemp[1] = x[1] + 0.01*gsl_ran_gaussian(r,1);
    xtemp[2] = x[2] + 1.0*gsl_ran_gaussian(r,1);
    
    double logH = loglikelihood(xtemp,C,N) - loglikelihood(x,C,N);
    
    double alpha = log(gsl_rng_uniform(r));
    
    if(logH>alpha)
    {
        x[0] = xtemp[0];
        x[1] = xtemp[1];
        x[2] = xtemp[2];
    }
    
    free(xtemp);
}
