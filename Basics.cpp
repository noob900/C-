#include <iostream>
using namespace std;

int main(){

int height, width;

cout<<"height";cin>>height;
cout<<"width";cin>>width;

for (int i = 0; i < height; i++){

  for (int j = 0; j < width; j++){
    cout<<"*";
  }
  cout<<endl;
}

} 