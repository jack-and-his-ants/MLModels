#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
using namespace std;
typedef struct compute_gradient_result{
    vector<double> dj_dw;
    double dj_db;
}compute_gradient_result;

typedef compute_gradient_result (*FuncPtr)(vector<vector<double>>&x, vector<double>&y,vector<double>& w,double b);


double array_multiplying(vector<double> &a1,vector<double> &a2){
    double result = 0;
    for(int i = 0; i <a1.size();i++){
        result += a1[i]*a2[i];
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////
double compute_cost(vector<vector<double>>&x, vector<double>&y,vector<double> &w,double b){
   
    size_t m = x.size();
    double cost = 0;
    double total_cost = 0;
    
    for(int i=0;i<m;i++){
        double f_wb = array_multiplying(x[i],w) + b;
        cost = cost + (f_wb - y[i])*(f_wb - y[i]);
    }
    
    total_cost = cost / (2 * m);
    return total_cost;
}
compute_gradient_result compute_gradient(vector<vector<double>>&x, vector<double>&y,vector<double>&w,double b){
    size_t m = x.size(); 
    size_t n = x[0].size();   
    vector<double> dj_dw;
    for(int i = 0;i<x[0].size();i++){
        dj_dw.push_back(0);
    }
    double dj_db = 0;
    
    // for i in range(m):                             
    //     err = (np.dot(X[i], w) + b) - y[i]   
    //     for j in range(n):                         
    //         dj_dw[j] = dj_dw[j] + err * X[i, j]    
    //     dj_db = dj_db + err                        
    // dj_dw = dj_dw / m                                
    // dj_db = dj_db / m
    for (size_t i=0;i<m;i++){
        double error = (array_multiplying(x[i],w) + b) - y[i];
        for(int j=0;j<n;j++){
            dj_dw[j] = dj_dw[j] + error*x[i][j];
        }
        dj_db = dj_db + error;
    }
    for(int i = 0;i<dj_dw.size();i++){
        dj_dw[i] /= m;
    }
    dj_db = dj_db / m ;
    
    compute_gradient_result r = {dj_dw,dj_db};
    return r;
}
compute_gradient_result gradient_descent(vector<vector<double>>&x,vector<double>& y,vector<double>& w_in,double b_in,double alpha,int num_iters,FuncPtr gradient_function){ 
    double b = b_in;
    vector<double> w;
    for(int i=0;i<w_in.size();i++){
        w.push_back(w_in[i]);
    }
    
    for(int i=0;i<(num_iters);i++){
        //# Calculate the gradient and update the parameters using gradient_function
        compute_gradient_result res = gradient_function(x, y, w , b);
        vector <double> dj_dw = res.dj_dw;      
        double dj_db = res.dj_db;
        //# Update Parameters using equation (3) above
        b = b - alpha * dj_db;
        for (int j = 0; j<w.size();j++){
            w[j]=w[j]-alpha*dj_dw[j];
        }                   

        //# Save cost J at each iteration
        if(i%5 == 0){
            cout << "Cost: "<< compute_cost(x, y, w, b);
            cout << " Iteration: " << i << endl;
        }
    }
    compute_gradient_result r = {w, b};
    return r;
}

int main(){
    vector<vector<double>> x_train = {{1.0,3.0},{2.0,1.0},{3.0,10.0},{4.0,0},{5.0,1}};
    vector<double> y_train = {250.0,262.0,840.0,418.0,540.0};
    double alpha = 0.03;
    vector<double> init_weights = {10,30};
    compute_gradient_result weights = gradient_descent( x_train , y_train , init_weights, 50.0 , alpha , 4000 , compute_gradient );
    for(int i=0;i<weights.dj_dw.size();i++){
        cout << " W" << i << ": " << weights.dj_dw[i];
    }
    cout  << " B: " << weights.dj_db << endl;

    cout << "prediction for 6,2: " << weights.dj_dw[0]*6.0 + weights.dj_dw[1]*2.0 + weights.dj_db << endl;
}