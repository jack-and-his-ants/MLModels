#include <iostream>
#include <vector>
#include <string>
using namespace std;

typedef struct compute_gradient_result{
    vector<double> dj_dw;
    double dj_db;
}compute_gradient_result;

typedef compute_gradient_result (*FuncPtr)(vector<vector<double>>&x, vector<double>&y,vector<double>& w,double b);

double array_multiplying(vector<double> &a1,vector<double> &a2);

double compute_cost(vector<vector<double>>&x, vector<double>&y,vector<double> &w,double b);

compute_gradient_result compute_gradient(vector<vector<double>>&x, vector<double>&y,vector<double>&w,double b);

compute_gradient_result gradient_descent(vector<vector<double>>&x,vector<double>& y,vector<double>& w_in,double b_in,double alpha,int num_iters,FuncPtr gradient_function);

