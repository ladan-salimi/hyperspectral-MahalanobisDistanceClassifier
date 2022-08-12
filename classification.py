#hyperspectral classification using MahalanobisDistanceClassifier
from spectral import *

#open img
img = open_image(r'C:\Users\ladan\Desktop\clustring\92AV3C.lan').load()
gt = open_image(r'C:\Users\ladan\Desktop\clustring\92AV3GT.GIS').read_band(0)
v = imshow(classes=gt)
#___________________________
#create a TrainingClassSet object
classes = create_training_classes(img, gt)
################
gmlc = MahalanobisDistanceClassifier(classes)
clmap = gmlc.classify_image(img)
v = imshow(classes=clmap)
gtresults = clmap * (gt != 0)
v = imshow(classes=gtresults)
#############
#to view errors:
gterrors = gtresults * (gtresults != gt)
v = imshow(classes=gterrors)   
##########
