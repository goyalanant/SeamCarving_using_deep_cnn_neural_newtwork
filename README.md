# SeamCarving_using_deep_cnn_neural_newtwork
  It is a project where we tried to improve seam carving technique in terms of speed and accuracy.

Note:- Energy map ef an image is the image which contains the edges of the objects(gray scale image).Obtained through sobel,scharr,laplacian and many more filters.
Overview:- To see what is project doing an overview video is uploaded above plz go download and watch
##Seam Carving:-
    It is a Image processing technique which is used to preserve the main objects in an image while cropping.
    
### Conventional Approach:-
  In conventional approach we uses greedy and dynamic programming to find the minimum energy vertical pixel seam and remove it.
  But in case of greedy approach is fast but the results are not satistfying.
  In case of DP it is too slow but outputs are comparitively good.
  
Bipartite Grpah is good in case of speed and accuracy.

Now In genreal approch we are having some predifined filters as discuss above which may not be good for particular image.
By using Deep CNN eural network we will genrelized the best filter and find the respective best energy map giving us best seam carved image. 

