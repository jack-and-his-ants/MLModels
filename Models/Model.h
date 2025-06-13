#pragma once
#include <iostream>
#include <vector>
#include <string>
using namespace std;


class Model {
    protected:
        string modelType;
    public:
        virtual void train(){};
        virtual double predict(){return 0;};
        friend std::ostream& operator<<(std::ostream& os, const Model&model);
        virtual void print(std::ostream& out) const = 0;
};