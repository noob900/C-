#include <iostream>
#include <list>
#include<vector>


class Bisection_method {
public:
    bool isPalindrome(long int x) {
        long int revnum = 0;
        long int temp=x;
        long int remainder=0;
        
         while(temp != 0) {
            remainder = temp % 10;
            revnum= revnum * 10 + remainder;
            temp /= 10;
          }
        if(revnum==x && x>=0){
            cout<<x<<" is palindrome";
            return true;
        }
        else{
             cout<<x <<" is not a palindrome";
            return false;
        }  
    };

    //The intermediate value theorem
    //f(a).f(b)<0 

    float a=0; 
    float b=0;
    float x=0; 
    float y=0;

    float f= 2^x+


};