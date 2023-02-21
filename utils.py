import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



def load_img( img_list):
    images=[]
    scaler = MinMaxScaler()
    for _, image_name in enumerate(img_list):    
            
        temp_image_t1ce=nib.load(image_name).get_fdata()
        temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
        temp_combined_images = temp_image_t1ce[...,np.newaxis]
        temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
        images.append(temp_combined_images)
    images = np.array(images)
    
    return(images)

def load_mask( mask_list):
    masks=[]
    for _, image_name in enumerate(mask_list): 
        
        temp_mask=nib.load(image_name).get_fdata()
        temp_mask=temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
        temp_mask = temp_mask[56:184, 56:184, 13:141]
        temp_mask= to_categorical(temp_mask, num_classes=4)
        
        masks.append(temp_mask)
    masks = np.array(masks)
    
    return(masks)



def imageTestLoader(img_list, mask_list, size=5):
    tests = []
    for i in np.random.choice(len(img_list), size, replace=False):
        image_path = img_list[i]
        mask_path = mask_list[i]

        x = load_img([image_path])

        x = np.moveaxis(x, -1, 1)
        x = np.squeeze(x)

        y = load_mask([mask_path])

        y = np.moveaxis(y, -1, 1)
        y = np.squeeze(y)
        y = np.moveaxis(y, 0, -1)
        y = np.argmax(y, axis=-1)

        tests.append((x[np.newaxis, np.newaxis, ...], x, y))
    return tests


def testImages(test_examples, model, slice=80):
    rows, cols = 2, min(len(test_examples), 5)
    plt.figure(figsize=(20, 10))

    for i, test in enumerate(test_examples):
        x, _, _ = test
        pred = model.predict(x)
        pred_mask= pred[0]

        pred_mask= np.squeeze(pred_mask)
        pred_mask=np.moveaxis(pred_mask,0,-1)
        pred_mask=np.argmax(pred_mask,axis=-1)
        
        plt.subplot(rows, cols, i+1)
        plt.imshow(pred_mask[:, :, slice])
        plt.title(f'Predicted mask {i}')

    for i, test in enumerate(test_examples):
        _, x_original, _ = test

        plt.subplot(rows, cols, i+6)
        plt.imshow(x_original[:, :, slice], cmap='gray')
        plt.title(f'Original mask {i}')

    plt.show()
