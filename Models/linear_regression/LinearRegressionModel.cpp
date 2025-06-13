#include <iostream>
#include <vector>
#include <string>
#include "LinearRegressionModel.h"
#include "../Model.h"
#include <random>
#include <chrono>
using namespace std;

LinearRegressionModel::LinearRegressionModel(int numParameters,double learningRate){
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());

    std::uniform_real_distribution<double> dist(1.0, 5000.0);
    for (int i=0;i<numParameters;i++){
        this->weights.push_back(dist(gen));
    }
    this->b=dist(gen);
    this->learningRate = learningRate;
    this->modelType = "Linear Regression Model";
    this->use = "Undefined";
}
double LinearRegressionModel::predict(vector<double>&data){
    int weightLength = this->weights.size();
    if(data.size()!=weightLength){
        cout << "Incorrect data!" << endl;
        return NAN;
    }
    double result = 0;
    for(int i=0;i<weightLength;i++){
        result += data[i]*this->weights[i];
    }
    result += this->b;
    return result;
}
double LinearRegressionModel::array_multiplying(const vector<double> &a1,const vector<double> &a2)const{
    double result = 0;
    for(int i = 0; i <a1.size();i++){
        result += a1[i]*a2[i];
    }
    return result;
}

double LinearRegressionModel::compute_cost(vector<vector<double>>&x, vector<double>&y)const{
    size_t m = x.size();
    double cost = 0;
    double total_cost = 0;
    
    for(int i=0;i<m;i++){
        double f_wb = array_multiplying(x[i],this->weights) + this->b;
        cost = cost + (f_wb - y[i])*(f_wb - y[i]);
    }
    
    total_cost = cost / (2 * m);
    return total_cost;
}
compute_gradient_result LinearRegressionModel::compute_gradient(vector<vector<double>>&x, vector<double>&y)const{
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
        double error = (array_multiplying(x[i],this->weights) + this->b) - y[i];
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


void LinearRegressionModel::train(vector<vector<double>> &x,vector<double> y,int numEpochs){
    for(int i=0;i<(numEpochs);i++){
        //# Calculate the gradient and update the parameters using gradient_function
        compute_gradient_result res = compute_gradient(x, y);
        vector <double> dj_dw = res.dj_dw;      
        double dj_db = res.dj_db;
        //# Update Parameters using equation (3) above
        b = b - this->learningRate * dj_db;
        for (int j = 0; j<this->weights.size();j++){
            this->weights[j]=this->weights[j]-learningRate*dj_dw[j];
        }                   

        //# Save cost J at each iteration
        if(i%5 == 0){
            cout << "Cost: "<< compute_cost(x, y);
            cout << " Iteration: " << i << endl;
        }
    }
}
// SETTERS
void LinearRegressionModel::setLearningRate(double nlr){
    this->learningRate = nlr;
}
void LinearRegressionModel::setUsePurpose(string & newstring){
    this->use = newstring;
}

// GETTERS
double LinearRegressionModel::getFreeDigit(){
    return this->b;
}

double LinearRegressionModel::getLearningRate(){
    return this->b;
}

vector<double> LinearRegressionModel::getWeights(){
    return this->weights;
}

void LinearRegressionModel::print(std::ostream& out)const{
    cout << "Linear Regression Model to use with " << this->use << endl;
}

LinearRegressionModel::~LinearRegressionModel(){
    cout << "Model deleted :)" << endl;
}