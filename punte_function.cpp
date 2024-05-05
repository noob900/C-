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
        list<string> Courses; 

     //constructor
    persondetails(std::string name,string address, string school){
        Fullname= name;
        Age=0;
        Address = address;
        School=school;

        cout<< "Full name of given person is " << Fullname<<endl; 
        cout<<Fullname<< " address is " << Address<<endl; 
        cout << Fullname << " studies in " << School<<endl;
    }

};

int main(){
    persondetails shishir("shishir pathak","lappeenranta", "LUT");
    shishir.Courses.push_back("Contorls");
    shishir.Courses.push_back("Machine Dynamics");

    persondetails Nima("Nima Palmu Sherpa","lappeenranta", "LAB");
}