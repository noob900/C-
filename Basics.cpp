#include <iostream>
#include <iomanip>
using namespace std;

int main(){

int height, width, space;
char symbol;

cout<<"height:";cin>>height;
cout<<"width:";cin>>width;
cout<<"symbol:";cin>>symbol;
cout<<"spacing:";cin>>space;

for (int i = 0; i < height; i++){

  for (int j = 0; j < width; j++){
    cout<<setw(space)<<symbol;
  }
  cout<<endl;
}

} 