#include <iostream>
using namespace std;

int main()
{

int number, revnumber; 
cout<<"enter atleast 2 digit number:"; cin >> number;

revnumber=number%10;
number=number/10; 

cout<<number;
cout<<revnumber;

} 