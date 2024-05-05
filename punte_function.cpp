#include <iostream>
#include <List>
using namespace std; 

class person{
    
    public:
        string fullname;
        int age; 
        char home; 
        string school; 
    private:
        string boyfriend;
};

int main(){
    person nima; 
    nima.fullname = "Nima palmu sherpa";

    cout<< "Full name of given person is:" << nima.fullname<<endl; 
    return 0;
}