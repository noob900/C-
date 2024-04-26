#include <iostream>
using namespace std;

int main()
{

int number, revnumber=0; 
cout<<"enter atleast 2 digit number:"; cin >> number; //123

do
{
cout<<number<<endl;
revnumber=(number%10+revnumber*10); // 32
number=number/10; // 1
 
} while(number!=0);

cout<< revnumber <<endl;

} 