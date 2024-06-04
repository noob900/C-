#include <iostream>
#include <list>
#include<vector>

using namespace std;


class sumtwonum{

public:

    vector<int> twoSum(vector<int>& nums, int target) {
        // split the array
        int sizeofvector=nums.size();
        int internalnumber;
        vector<int> finalvector;

        for(int i=0;i<sizeofvector;i++){
            for(int j=i+1;j<sizeofvector;j++){
                internalnumber=nums[i]+nums[j];

                if(internalnumber==target){
                    finalvector={i,j};
                    
                    cout<<"sum of " << nums[i]<< " and "<< nums[j] << "is " << internalnumber<<endl;
                } 
            }
        }

        return {};
    }
};
class Palindromenum {
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
};
int main() { 

    // sum of two num to reach a target
    sumtwonum Sum;
    vector<int> nums = {2,5,5,11};
    int target = 9;
    Sum.twoSum(nums,target);

    // palindrome 
    Palindromenum palindrome; 
    palindrome.isPalindrome(123321);
};