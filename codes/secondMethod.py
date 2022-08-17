import sys
import os
import cv2

def secondMethod(path1,path2):  
  images1 = [file for file in sorted(os.listdir(path1)) if file.endswith(('png'))]
  images2 = [file for file in sorted(os.listdir(path2)) if file.endswith(('png'))]
  count=0
  for i in range(len(images1)):      
      imb = cv2.imread(path1+'/'+images1[i])
      imf = cv2.imread(path2+'/'+images2[i], 1)      
      #imb[imb<30]=0
      imf[imf<25]=0
      imtot=imb+imf
      cv2.imwrite('/content/results/MySecondMod/frame%d.png' % count, imtot)    
      count += 1
  print('done')

a=(sys.argv)  
if a[1]=='001':
  path1='/content/results/Background/MyG_iter12000/background'
else: path1='/content/results/Background/EGVSR_iter420000/background'
path2='/content/results/Foreground/EGVSR_iter420000/foreground/'
print(path1)
secondMethod(path1,path2)      
