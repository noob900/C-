#include <iostream>
#include <list>
#include<vector>

using namespace std; 


template<typename any>
class sizecheck{

    public:
    int totalsize;

    sizecheck(vector<any> elementsofarray ){
        totalsize=elementsofarray.size();
        cout<<"size of the array is: "<< totalsize<<endl;
     }
    

};
class words{

private:
    vector<char> internalsavedword;

public:
    // this is a constructor
    words(string inputword){

    //saves the input word in an array. 
    vector<char> savedword;
    for(int i = 0; i < inputword.length(); i++){
        savedword.push_back(inputword[i]);
    }  
    // prints the array list  
    for(const char &savedword : savedword){
        cout<<savedword;
    } 
    cout<<endl;
    sizecheck wordsizecheck(savedword);
    internalsavedword=savedword;
    };

    // this is a Method to reverse the word
    void reverseword(){
        vector<char> reversedword;
        for(int i = internalsavedword.size(); i >= 0; i--){
            reversedword.push_back(internalsavedword[i]);
            cout<<internalsavedword[i]; 
        }
    }

    void orders(vector<int> ordernumber){
        
      vector<char> orderletter=internalsavedword;
      for(int i = 0; i < ordernumber.size(); i++){
        int asciiValue = static_cast<int>(orderletter[i]);
        asciiValue=asciiValue+ordernumber[i];
        orderletter[i]= static_cast<char>(asciiValue);
        cout<<orderletter[i]<<"---"<<asciiValue<<endl;
      }  
  }
};

class number{
    private: 
        vector<int> integerlist;
        int numberofelement;
        int internalnumber;
        

    public:

    number(vector<char> integers_list){
        for(const auto &intergers : integerlist){
            cout<< intergers << "," ; 
        }
        cout<<endl;
        
        sizecheck numbersizecheck(integers_list);
        int repetition;

        for(int i=0; i < numbersizecheck.totalsize; i++){
            repetition=0;
            internalnumber=integers_list[i]; //pointer
            for(int j=0;j<numbersizecheck.totalsize; j++) {
                if(internalnumber==integers_list[j]){
                    repetition++;
                }
            } 

            if(repetition>1){
                cout<<"total repetition of " << internalnumber<< "is: " << repetition<<endl;
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


int main(){

    words tweakword("hello"); 
    tweakword.orders({-1,2,-5,-5,-8});
    number numbers({1,2,3,1,3,4,5,1,2,1,3,5,1});

}