import albumentations as alb
import cv2
import os
import xmltodict
import pprint

labels=[
    'r', # resistor
    'b', # inductor
    'c', # capacitor
    'v', # voltage source
    'd', # diode
    'l'  # lamp
]
label2color={
    'r':(255, 255, 102), # yellow
    'b': (102, 255, 51), # light green
    'c': (102, 204, 255), # light blue
    'v': (255, 102, 255), # pink
    'd': (255, 51, 0), # red
    'l': (51, 51, 255) # dark blue
}
augmentor=alb.Compose([
    alb.HorizontalFlip(p=0.5),
    alb.VerticalFlip(p=0.5),
    alb.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(255,255,255),p=1,crop_border=False)
],bbox_params=alb.BboxParams(format='pascal_voc'))


img=cv2.imread(os.path.join('data','test_alb','img3.jpg'))
file = open(os.path.join('data','test_alb','img3.xml'),'rb')
labels=xmltodict.parse(file)
pp = pprint.PrettyPrinter(indent=1)
# pp.pprint(labels)
bboxes=[]
for obj in labels['annotation']['object']:
    #get coordinates
    coords=obj['bndbox'].values()
    coords=[int(i) for i in coords]
    bboxes.append(coords+[obj['name']])

# print(bboxes)

augmented=augmentor(image=img, bboxes=bboxes)
aug_img=augmented['image']
# pp.pprint(augmented['bboxes'])
for box in augmented['bboxes']:
    color=label2color[box[4]]
    coordinates=[(int(box[0]),int(box[1])),(int(box[2]),int(box[3]))]
    cv2.rectangle(aug_img,*coordinates, color, 1)


cv2.imshow('augmented image', aug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: augment all images, save them and generate xml pascal voc files
# TODO: refactor code