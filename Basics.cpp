#include <iostream>
using namespace std;
int charfigure();
int revnum();

int main(){
    charfigure();
    revnum();
};

int revnum(){
    int number, revnumber=0; 
    cout<<"enter atleast 2 digit number:"; cin >> number; 

    do{
      cout<<number<<endl;
      revnumber=(number%10+revnumber*10); 
      number=number/10; 
  
    } while(number!=0);

cout<< revnumber <<endl;
};


int charfigure(){
    int height, width;
    cout<<"height";cin>>height;
    cout<<"width";cin>>width;

  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
    cout<<"*";
  }
  cout<<endl;
}
};
 
