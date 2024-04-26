#include <iostream>
using namespace std;

int main()
{

int number, revnumber=0; 
cout<<"enter atleast 2 digit number:"; cin >> number; 

do
{
cout<<number<<endl;
revnumber=(number%10+revnumber*10); 
number=number/10; 
 
} while(number!=0);

cout<< revnumber <<endl;

} 
