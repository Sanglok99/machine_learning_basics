import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


train_data=pd.read_csv('Train_data.csv')
val_data=pd.read_csv('Validation_Data.csv')
test_data=pd.read_csv('Test_Data.csv')


def Write_Image_Sizes(filenames,storage_file):
    """
    Takes the File names, writes the width and height of images in csv along with file names
    """
    store_file=open(storage_file,'w+')
    store_file.write("ImageName,Height,Width")
    store_file.write("\n")
    cnt=0
    for file in filenames:
        cv_img=cv2.imread(file)
        #img.shape gives (img_height,img_width,img_channel)
        store_file.write(str(file)+","+str(cv_img.shape[0])+","+str(cv_img.shape[1]))
        store_file.write("\n")
        cnt+=1
        if cnt%10000==0:
            print("Processed Images: ",cnt)
    store_file.close()

train_image_names = list(train_data['ImageName'].values)
val_image_names = list(val_data['ImageName'].values)
test_image_names = list(test_data['ImageName'].values)

Write_Image_Sizes(train_image_names, 'Train_image_sizes.csv')
Write_Image_Sizes(val_image_names, 'Validation_image_sizes.csv')
Write_Image_Sizes(test_image_names, 'Test_image_sizes.csv')

train_img_size = pd.read_csv('Train_image_sizes.csv')
val_img_size = pd.read_csv('Validation_image_sizes.csv')
test_img_size = pd.read_csv('Test_image_sizes.csv')

train_img_size.describe()
val_img_size.describe()
test_img_size.describe()

print("Train Images Height 90 percentile :", np.percentile(train_img_size['Height'].values, 90))
print("Train Images Height 99 percentile :", np.percentile(train_img_size['Height'].values, 99))
print("Train Images Width 90 percentile :", np.percentile(train_img_size['Width'].values, 90))
print("Train Images Width 99 percentile :", np.percentile(train_img_size['Width'].values, 99))
print("=" * 60)
print("Validation Images Height 90 percentile :", np.percentile(val_img_size['Height'].values, 90))
print("Validation Images Height 99 percentile :", np.percentile(val_img_size['Height'].values, 99))
print("Validation Images Width 90 percentile :", np.percentile(val_img_size['Width'].values, 90))
print("Validation Images Width 99 percentile :", np.percentile(val_img_size['Width'].values, 99))
print("=" * 60)
print("Test Images Height 90 percentile :", np.percentile(test_img_size['Height'].values, 90))
print("Test Images Height 99 percentile :", np.percentile(test_img_size['Height'].values, 99))
print("Test Images Width 90 percentile :", np.percentile(test_img_size['Width'].values, 90))
print("Test Images Width 99 percentile :", np.percentile(test_img_size['Width'].values, 99))

for i in range(10):
    print("Train Images Width "+str(90+i)+ " percentile :",np.percentile(train_img_size['Width'].values,90+i))
print("="*60)
for i in range(10):
    print("Validation Images Width "+str(90+i)+ " percentile :",np.percentile(val_img_size['Width'].values,90+i))
print("="*60)
for i in range(10):
    print("Test Images Width "+str(90+i)+ " percentile :",np.percentile(test_img_size['Width'].values,90+i))


def cdf_image_widths(label_len):
    """
    Takes a list of image widths as input and Plots CDF of image widths
    """
    plt.figure(figsize=(10,6))
    count_labels=np.array(label_len)
    counts, bin_edges = np.histogram(count_labels, bins=8,
                                 density = True)
    pdf = counts/(sum(counts))
    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel('Width of Images',fontsize=10)
    plt.ylabel('CDF',fontsize=10)
    plt.title('CDF Plot of Image Width',fontsize=12)
    plt.show()

cdf_image_widths(train_img_size['Width'].values)
cdf_image_widths(val_img_size['Width'].values)
cdf_image_widths(test_img_size['Width'].values)
