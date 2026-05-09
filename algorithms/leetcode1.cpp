#include <iostream>
#include <list>
#include<vector>
#include<map>
#include<unordered_map>

using namespace std;

class romantoint{
public:
    unordered_map<char, int> mp = {
        {'I', 1},
        {'V', 5},
        {'X', 10},
        {'L', 50},
        {'C', 100},
        {'D', 500},
        {'M', 1000}
    };

    int romanToInt(string s) {
        int sum = 0;
        for (int i = 0; i < s.length(); i++) {
            if (mp[s[i]] < mp[s[i + 1]]) {
                sum -= mp[s[i]];
            } else {
                sum += mp[s[i]];
            }
        }
        cout<<sum;
        return sum;
    }
};
class longestprefix {
public:
    map<string, vector<char>> mp = {};
    vector<vector<char>> letters = {};

    string longestCommonPrefix(vector<string> strs) {
        for (const auto &str : strs) {
            vector<char> tempcollecttor; // Initialize a new collector for each string
            for (int i = 0; i < str.length(); i++) {
                tempcollecttor.push_back(str[i]);
            }
            mp[str] = tempcollecttor; // Insert the string and its characters into the map
        }
        for (const auto &pair : mp) {
            for(int i=0;i<pair.second.size();i++){

            }
            
        }
        
        
        

        // Print the map
        for (const auto &pair : mp) {
            
            cout << "Key: " << pair.first << ", Value: ";
            for (const auto &ch : pair.second) {
                cout << ch << " ";
            }
            cout << endl;
        }

        // Returning an empty string as a placeholder for longestCommonPrefix functionality
        return "";
    }
};

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

    //Roman to integers
    romantoint romantoint;
    romantoint.romanToInt("DCVII");


};