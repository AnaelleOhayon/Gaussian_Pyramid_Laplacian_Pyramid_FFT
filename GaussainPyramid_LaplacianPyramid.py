# Shivam Chourey
# shivam.chourey@gmail.com

import cv2
import numpy as np

# Weights of the Gaussian Blur
a = np.array([1/16, 1/4, 3/8, 1/4, 1/16])
b = np.array([[1/16], [1/4], [3/8], [1/4], [1/16]])
Weights = a*b

Iteration = [-2, -1, 0, 1, 2]


def GaussianPyramidGenerator(Input):

   w, h = Input.shape[:2]

   GaussianPyramidImage = np.zeros((int(w/2),int(h/2)))

   for index in range(int(w/2)):
      for itr in range(int(h/2)):
        for var in range(5):
           for cor in range(5):
             GaussianPyramidImage[index][itr] += Weights[var][cor]*Input[2*index - var][2*itr - cor]
             
   return GaussianPyramidImage
   
   
def LaplacianPyramidGenerator(Input):

   w, h = Input.shape[:2]

   LaplacianPyramidImage = np.zeros((int(w*2),int(h*2)))

   for index in range(int(w*2)):
      for itr in range(int(h*2)):
             LaplacianPyramidImage[index][itr] += Input[int(index/2)][int(itr/2)]
             
   return LaplacianPyramidImage
   
   
def PyramidImageForFiveLevel(ImageList):
    DisplayImage = np.ones((w, int(1.5*h)))
    widthNum = 0
    heightNum = 0
    for index in range(len(ImageList)):
    
        ImageDisp = ImageList[index]
        tmpW, tmpH = ImageDisp.shape[:2]
        
        # Decimation factor of 2
        # Image size is halved every iteration
        if(index != 0):
           heightNum = h
        
        if(index == 2):
           widthNum = int(w/2)
           
        if(index == 3):
           widthNum = int(w/2 + w/4)
        
        if(index == 4):
           widthNum = int(w/2 + w/4 + w/8)
           
        if(index == 5):
           widthNum = int(w/2 + w/4 + w/8 + w/16)
            
        for i in range(tmpW):
           for j in range(tmpH):
              DisplayImage[i+widthNum][j+heightNum] = ImageDisp[i][j]/256
    
    return DisplayImage

OriginalName = "ENTER FILENAME HERE"
Original = cv2.imread(OriginalName ,cv2.IMREAD_GRAYSCALE)
w, h = Original.shape[:2]

fft_o = np.fft.fft2(Original)
fft_oname = OriginalName[:-4]+ "_fft.jpg"
cv2.imwrite(fft_oname, np.abs(fft_o))

current = Original

GaussianImages = [Original]
GaussianFFTImages = []
LaplacianImages = []
LaplacianFFTImages = []

for cycle in range(5):

   # Build Gaussian Pyramid
   print("Started Gaussian Pyramid")
   result = GaussianPyramidGenerator(current)

   filename = OriginalName[:-4]+"_GP_Level_"+ str(cycle+1) +".jpg"
   cv2.imwrite(filename, result)
   GaussianImages.append(result)
   
   #Build Laplacian Pyramid
   print("Started Laplacian Pyramid")
   Laplacian = LaplacianPyramidGenerator(result)
   
   # Get the difference
   print("Get the Difference Image")
   Cw, Ch = current.shape[:2]
   Lw, Lh = Laplacian.shape[:2]
   
   # Image resize in cases where width or height aren't factor of 2^5
   if(Cw != Lw or Ch != Lh):
      Laplacian = cv2.resize(Laplacian, (Ch, Cw))
   
   Diff = current - Laplacian

   diffname = OriginalName[:-4]+"_Diff_Level_"+ str(cycle+1) +".jpg"
   cv2.imwrite(diffname, Diff) 
   LaplacianImages.append(Diff)

   
   # Get the Fourier transform of each image
   fft_g = np.fft.fft2(result)
   GaussianFFTImages.append(np.abs(fft_g))
   fft_gname = OriginalName[:-4]+"_GP_Level_"+ str(cycle+1)+ "_fft" +".jpg"
   cv2.imwrite(fft_gname, np.abs(fft_g))

   fft_d = np.fft.fft2(Diff)
   LaplacianFFTImages.append(np.abs(fft_d))
   fft_dname = OriginalName[:-4]+"_Diff_Level_"+ str(cycle+1) + "_fft" +".jpg"
   cv2.imwrite(fft_dname, np.abs(fft_d))
   
   # Update condition of the loop
   current = result
   #End of loop

GDisplayImage = PyramidImageForFiveLevel(GaussianImages)
GFFTDisplayImage = PyramidImageForFiveLevel(GaussianFFTImages)
LDisplayImage = PyramidImageForFiveLevel(LaplacianImages)
LFFTDisplayImage = PyramidImageForFiveLevel(LaplacianFFTImages)
      
cv2.imshow( "Gaussian Pyramid", GDisplayImage )
GPName = OriginalName[:-4]+"_GaussianPyramid.jpg"
cv2.imwrite(GPName, GDisplayImage*256)
cv2.imshow( "Gaussian FFT Pyramid", GFFTDisplayImage )
GPFFTName = OriginalName[:-4]+"_GaussianPyramid_FFT.jpg"
cv2.imwrite(GPFFTName, GFFTDisplayImage*256)

cv2.imshow( "Laplacian Pyramid", LDisplayImage )
LPName = OriginalName[:-4]+"_LaplacianPyramid.jpg"
cv2.imwrite(LPName, LDisplayImage*256)
cv2.imshow( "Laplacian FFT Pyramid", LFFTDisplayImage )
LPFFTName = OriginalName[:-4]+"_LaplacianPyramid_FFT.jpg"
cv2.imwrite(LPFFTName, LFFTDisplayImage*256)


cv2.waitKey(0)

cv2.destroyAllWindows()


print("Process completed") 

  




