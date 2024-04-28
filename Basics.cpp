#include <iostream>
#include <iomanip>
using namespace std;

void  triangle(int height, int width, int space, char symbol){

for (int i = 0; i <= height; i++){
  for (int j = 0; j<=i; j++){
    cout<<setw(space)<<symbol;
  }
  cout<<endl;
}

for (int i = width; i >= 1; i--){
  for (int j = 1; j<=i; j++){
    cout<<setw(space)<<symbol;
  }
  cout<<endl;
}
}

int main(){
  triangle(2,2,2,2);
}


