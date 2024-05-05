#include <iostream>
#include <list>
using namespace std; 

class persondetails{
    
    //attributes

    private:
        string boyfriend;
        float salary; 

    public:
        string Fullname;
        int Age; 
        string Address; 
        string School; 

     //constructor
    persondetails(std::string name,string address, string school){
        Fullname= name;
        Age=0;
        Address = address;
        School=school;
    }


    
};

int main(){
    persondetails shishir("shishir","lappeenranta", "LUT");

    cout<< "Full name of given person is:" << shishir.Fullname<<endl; 

}