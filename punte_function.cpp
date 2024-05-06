#include <iostream>
#include <list>
#include<vector>
using namespace std; 

class words{

private:
    vector<char> internalsavedword;
    int internalnumberofelement;

public:
    // this is a constructor
    words(string inputword){

    //saves the input word in an array. 
    vector<char> savedword;
    for(int i = 0; i < inputword.length(); i++){
        savedword.push_back(inputword[i]);
    }  
    // prints the array list  
    int numberofelement=savedword.size(); 
    for(const char &savedword : savedword){
        cout<<savedword;
    } 
        
    cout<<endl<<numberofelement<<endl;
    internalsavedword=savedword;
    internalnumberofelement=numberofelement;
    };

    // this is a Method to reverse the word
    void reverseword(){
        vector<char> reversedword;
        for(int i = internalnumberofelement; i >= 0; i--){
            reversedword.push_back(internalsavedword[i]);
            cout<<internalsavedword[i]; 
        }
    }

    void orders(vector<int> ordernumber){
        
    
      vector<char> orderletter=internalsavedword;
      for(int i = 0; i < ordernumber.size(); i++){
          int asciiValue = static_cast<int>(orderletter[i]);
          
          switch(ordernumber[i]){
                case 1: 
                    asciiValue=asciiValue+1;
                    orderletter[i]= static_cast<char>(asciiValue);
                    cout<<orderletter[i]<<"---"<<asciiValue<<endl;
                    break; 
              
                case -1: 
                    asciiValue=asciiValue-1;
                    orderletter[i]= static_cast<char>(asciiValue);
                    cout<<orderletter[i]<<"---"<<asciiValue<<endl;
                    break;
          }

      }  
  }
};


class persondetails{
    
    //attributes

    private:
        string Fullname;
        int Age; 
        string Address; 
        string School; 
        list<string> Courses; 
        string boyfriend;
        float salary; 

    public:

     //constructor
    persondetails(std::string name,string address, string school){
        Fullname= name;
        Age=0;
        Address = address;
        School=school;
    }

    void get_info(){
        cout<< "Full name of given person is " << Fullname<<endl; 
        cout<<Fullname<< " address is " << Address<<endl; 
        cout << Fullname << " studies in " << School<<endl;
        for (string Courses: Courses){
            cout<<Courses<<endl;
        };
    }

    void addcourses(string coursename){
        Courses.push_back(coursename);
    }

    void coursesize(){
         cout<< Courses.size() << endl;
    }
};

void printdata(void* ptr, char type){
    switch(type){

    case 'i': cout << *((int*) ptr);
    case 'c': cout << *((char*) ptr);  
    }
   
}

int main(){

    words tweakword("hello"); 
    tweakword.orders({1,1,1,1,1});

}