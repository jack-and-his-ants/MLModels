#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "Models/Model.h"
#include "Models/linear_regression/LinearRegressionModel.h"
using namespace std;


int main(){
    vector<vector<double>> x_train = {{1.0,3.0},{2.0,1.0},{3.0,10.0},{4.0,0},{5.0,1}};
    vector<double> y_train = {260.0,260.0,810.0,410.0,560.0};
    double alpha = 0.03;
    LinearRegressionModel*newModel = new LinearRegressionModel(2,alpha);
    newModel->train(x_train,y_train,4000);
    vector<double> y_test = {6,1,2};
    double prediction = newModel->predict(y_test);
    cout << "predykcja: "<< prediction << endl;
    delete newModel;
    

}