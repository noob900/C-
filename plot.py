import matplotlib.pyplot as plt
import numpy as np

x = int(input("Enter the starting number: "))
y = int(input("Enter the ending number: "))
prime_number=[]
for i in range(x,y+1):
  if i>1:
    for j in range(2,i):
      if i%j==0:
        break
    else:
      prime_number.append(i)

print(prime_number)


ypoints = prime_number
plt.plot(ypoints)
plt.show()
 

       

