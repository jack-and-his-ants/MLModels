#include "Model.h"
#include <iostream>
using namespace std;


ostream& operator<<(std::ostream& os, const Model& model) {
    model.print(os);
    return os;
}