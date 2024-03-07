import pandas as pd
import cv2
from datetime import datetime

def Extract_image_names(file_path,number):
    """
    Takes the file path of images annotation txt file with the number of images names to be extracted
    and returns the list of file names having label length <=12
    """
    with open(file_path) as f:
        file_names=f.readlines()
        f.close()
        count=0
        img_names=[]
        for file in file_names:
            _,label,_=file.split('_')
            if len(label)>=4 and len(label)<=12:
                img_names.append(file)
                count+=1
            if count==number:
                break
        images_names=['SynthImageDataset'+x.strip() for x in img_names]
        return images_names
    

def clean_file_names(file_names):
    clean_files=[]
    for file in file_names:
        main_folder,img_loc,extension=file.split('.')
        #Removing the image number at the end
        extension,_=extension.split(' ')
        img_file=main_folder+img_loc+'.'+extension
        clean_files.append(img_file)
    return clean_files


def extract_ground_truth(files):
    """
    Given the file names of images, extracts the Ground Truth Values and returns a list of Ground Truth Labels in All Capitals
    """
    txt_labels=[]
    for file in files:
        folder,ground_truth,image=file.split('_')
        ground_truth=ground_truth.upper()
        txt_labels.append(ground_truth)
    return txt_labels       

def img_store_single_channel(destination_folder,files):
    """
    Takes the images in a folder, distination folder path and 
    converts the image to single channel gray scale,
    stores the image in the destination folder and returns image destination list
    """
    start=datetime.now()
    destination_list=[]
    count=1
    for file in files:
        #Removing the extra folder structures
        _,_,_,Name=file.split('/')
        _,img,_=Name.split('_')
        destination=destination_folder+str(count)+'_'+img+'.jpg'
        cv_img=cv2.imread(file)
        #So extracting image from any 1 channel gives a single channel Grayscale image
        cv_img_sc=cv_img[:,:,1]
        cv2.imwrite(destination,cv_img_sc)
        destination_list.append(destination)
        count+=1
        if count%1000==0:
            print("Processed Images: ",count)
    print('Time Taken for Processing: ',datetime.now() - start)
    return destination_list

# x numbers of train data labeling
train_images= Extract_image_names('SynthImageDataset/annotation_train.txt',50000)
train_cleaned=clean_file_names(train_images)
train_data=pd.DataFrame({'ImageName':train_cleaned})
train_ground_truths=extract_ground_truth(train_cleaned)
train_data['Labels']=train_ground_truths    
train_data.to_csv('Train_data.csv')

# y numbers of validation data labeling
validation_images=Extract_image_names('SynthImageDataset/annotation_val.txt',5000)
val_cleaned=clean_file_names(validation_images)
val_data=pd.DataFrame({'ImageName':val_cleaned})
val_ground_truths=extract_ground_truth(val_cleaned)
val_data['Labels']=val_ground_truths
val_data.to_csv('Validation_data.csv')

# z numbers of test data labeling
test_images=Extract_image_names('SynthImageDataset/annotation_test.txt',5000)
test_cleaned=clean_file_names(test_images)
test_data=pd.DataFrame({'ImageName':test_cleaned})
test_ground_truths=extract_ground_truth(test_cleaned)
test_data['Labels']=test_ground_truths
test_data.to_csv('Test_data.csv')

#save train data into 'Train_data/' directory with convert to single channel
train_data=pd.read_csv('Train_data.csv')
train_data.drop(['Unnamed: 0'],axis=1,inplace=True)
train_files=train_data['ImageName'].values
train_dest=img_store_single_channel('Train_data/',train_files)
#Updating Train Dataframe with new destination file paths
train_data['ImageName']=train_dest
train_data.to_csv('Train_Final.csv')

#save test data into 'Test_data/' directory with convert to single channel
test_data=pd.read_csv('Test_data.csv')
test_data.drop(['Unnamed: 0'],axis=1,inplace=True)
test_files=test_data['ImageName'].values
test_dest=img_store_single_channel('Test_data/',test_files)
#Updating Test Dataframe with new destination file paths
test_data['ImageName']=test_dest
test_data.to_csv('Test_Final.csv')

#save validation data into 'Val_data/' directory with convert to single channel
val_data=pd.read_csv('Validation_data.csv')
val_data.drop(['Unnamed: 0'],axis=1,inplace=True)
val_files=val_data['ImageName'].values
val_dest=img_store_single_channel('Val_data/',val_files)
#Updating Validation Dataframe with new destination file paths
val_data['ImageName']=val_dest
val_data.to_csv('Validation_Final.csv')
