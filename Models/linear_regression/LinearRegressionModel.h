#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "../Model.h"
using namespace std;

typedef struct compute_gradient_result{
    vector<double> dj_dw;
    double dj_db;
}compute_gradient_result;

class LinearRegressionModel : public Model {
    private:
        string use;
        vector<double> weights;
        double b;
        double learningRate;
        double array_multiplying(const vector<double> &a1,const vector<double> &a2)const;
        compute_gradient_result compute_gradient(vector<vector<double>>&x, vector<double>&y)const;
        double compute_cost(vector<vector<double>>&x, vector<double>&y)const;

    public:
        LinearRegressionModel(int numParameters,double learningRate);
        void train(vector<vector<double>> &x,vector<double> y,int numEpochs);
        double predict(vector<double> &data);

        bool save();

        ~LinearRegressionModel();


        // SETTERS
        void setLearningRate(double nlr);
        void setUsePurpose(string & newstring);

        // GETTERS
        double getLearningRate();
        vector<double>getWeights();
        double getFreeDigit();
        

        // PRINT METHODS
        void print(std::ostream& out) const override;


};
