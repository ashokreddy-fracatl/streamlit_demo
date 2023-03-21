import streamlit as st
import os
import json
import yaml
import numpy as np
import pandas as pd
import warnings
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils import data_augmentation as da, service as svc
from utils.datasets import BuildingsDataset
from PIL import Image
from rectangle_from_contour import contour_find
from geo_co_ord_conversion import convert_2_geo_co_ord
import cv2

def boundary_coords(geo_json_path):
    f = open(geo_json_path)
    data = json.load(f)
    features = data['features']
    long_min  = 1e6
    long_max  = -1e6
    lat_min  = 1e6
    lat_max  = -1e6
    
    for i in range(len(features)):
        #print('--------------------------')
        # #print(len(features[i]['geometry']['coordinates'][0]))
        # 
        for k in range(len(features[i]['geometry']['coordinates'][0])):
            co_ord = features[i]['geometry']['coordinates'][0][k]
            #print('---------------')
            #print(co_ord)
            #print(co_ord[0],co_ord[1])
            #print(typ)
            if isinstance(co_ord, list) and len(co_ord) == 2:
                print('yes')
                if(co_ord[0]<long_min):
                    long_min = co_ord[0]
                if(co_ord[0]>long_max):
                    long_max = co_ord[0]
                if(co_ord[1]<lat_min):
                    lat_min = co_ord[1]
                if(co_ord[1]>lat_max):
                    lat_max = co_ord[1]
                
                
    #print('long_min,long_max',long_min,long_max)
    #print('lat_min,lat_max',lat_min,lat_max)
    
    return long_min,long_max,lat_min,lat_max

#from unet import UNet  # assuming you have defined your UNet model in a separate file called unet.py
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model weights

config_path = '/mnt/nfshome2/FRACTAL/ashok.reddy/Ashok/projects/flood_detection/roof_segmentation/rooftop_damage_assessment_models/roof_semseg/config/streamlit_app.yml'
with open(os.path.join('config/', config_path),encoding='utf-8') as config_load:
    config = yaml.safe_load(config_load)
# Get UNet model
net =  config['parameters']['NET'].split('/')
Net = getattr(__import__(f"net.{net[0]}",globals(),locals(),[net[1]],0), net[1])

geojson_dir_path = config['paths']['geo_json_dir']

class_csv_path = config['paths']['class_csv_path']
class_dict = pd.read_csv(class_csv_path)
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r','g','b']].values.tolist()

# Useful to shortlist specific classes in datasets with large number of classes
select_classes = config['parameters']['SELECT_CLASSES'].split(",")

# Get RGB values of required classes #black for background and white for object of interest
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]


best_model = Net(class_names)
PP_PARAMS = config['parameters']['PREPROCESSING'].split(',')
PREPROCESSING_FN = smp.encoders.get_preprocessing_fn(PP_PARAMS[0], PP_PARAMS[1])
best_model.load_state_dict(torch.load('/mnt/nfshome2/FRACTAL/ashok.reddy/Ashok/projects/flood_detection/roof_segmentation/rooftop_damage_assessment_models/roof_semseg/log/spacenet8_log/best_model.pth'))
best_model.to(DEVICE)
best_model.eval()

x_test_dir = config['paths']['x_test_dir']
for file in os.listdir(x_test_dir):
    os.remove(os.path.join(x_test_dir,file))


# Define Streamlit app
st.title('Roof top Detection')
uploaded_file = st.file_uploader('Choose an image file')
if uploaded_file is not None:
    #print('---------------------')
    #print('saving image file')
    image = Image.open(uploaded_file)
    image_input_path = os.path.join(x_test_dir,uploaded_file.name)
    image.save(image_input_path)
    list1 =  uploaded_file.name.split('.')[0].split('_')
    json_file_name = list1[1]+'_'+list1[2]+'_'+list1[3]+".geojson"
    geo_json_path = os.path.join(geojson_dir_path,json_file_name)
    print('geo_json_path',geo_json_path)
    #st.image(image, caption='Uploaded Image', use_column_width=True)

#print('------------------')
#print(len(os.listdir(x_test_dir)))
if(len(os.listdir(x_test_dir))!=0):
    y_test_dir = x_test_dir #no importance to x_test_dir
    #The long and lat values of given image
    long_min,long_max,lat_min,lat_max = boundary_coords(geo_json_path)
    print('------------------')
    print(long_min,long_max,lat_min,lat_max)
    # create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
    test_dataset = BuildingsDataset(
        x_test_dir,
        y_test_dir,
        augmentation=da.get_validation_augmentation(),
        preprocessing=da.get_preprocessing(PREPROCESSING_FN),
        class_rgb_values=select_class_rgb_values,
    )

    test_dataloader = DataLoader(test_dataset)

    test_dataset_vis = BuildingsDataset(
        x_test_dir, y_test_dir,
        augmentation=da.get_validation_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )
    #x_tensor = torch.tensor(np.float32(np.random.rand(1,3,1000,1000))).to(DEVICE)
    with torch.no_grad():
        for idx,test_sample in enumerate(test_dataset):
            image, gt_mask = test_sample
            image_vis = svc.crop_image(test_dataset_vis[idx][0].astype('uint8'))

            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        with st.spinner('Predicting...'):
            #x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            #mask = best_model(x_tensor).squeeze(0)
            #mask = best_model(x_tensor)
            #image = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            #image = torch.from_numpy(image).to(DEVICE)
            #mask = predict(image)
            pred_mask = best_model(x_tensor)
            #print(type(pred_mask))
            #print(pred_mask)
            #print(pred_mask.size())
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
            # Convert pred_mask from `CHW` format to `HWC` format
            pred_mask = np.transpose(pred_mask,(1,2,0))
            print(type(pred_mask))
            #print(pred_mask.size())
            # Get prediction channel corresponding to building
            pred_building_heatmap = pred_mask[:,:,select_classes.index('building')]
            print(type(pred_building_heatmap))
            print(pred_building_heatmap.shape)
            #print(pred_building_heatmap)
            #print(pred_building_heatmap.size())
            pred_mask = svc.crop_image(
                svc.colour_code_segmentation(
                    svc.reverse_one_hot(pred_mask),
                    select_class_rgb_values
                )
            )
            print(type(pred_mask))
            print(pred_mask)
            print(pred_mask.shape)
            masked_original = svc.overlay_mask(cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR),pred_mask)
            #masked_original = cv2.putText(masked_original, 'test', (500,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            #image_path_out = os.path.join(x_test_dir, "output.png")
            #cv2.imwrite(image_path_out,masked_original)
            mask_path_out = os.path.join(x_test_dir, "mask.png")
            cv2.imwrite(mask_path_out,pred_mask)
            #image_out = Image.open(image_path_out)
            
        #st.image(image_out, caption='Predicted Mask', use_column_width=True)

    
    st.write(f"The co ordinates of {uploaded_file.name}")
    st.write(f"longitude   [{round(long_min,4)} to {round(long_max,4)}]")
    st.write(f"latitude  [{round(lat_min,4)} to {round(lat_max,4)}]")



    #finding the rectangular co-ordinates
    rect_segments_dict = contour_find(mask_path_out)
    for key in rect_segments_dict.keys():
        rect_each = rect_segments_dict[key]
        mid_point = (int((rect_each[0]+rect_each[2])/2),int((rect_each[1]+rect_each[3])/2))
        masked_original = cv2.putText(masked_original, str(key), mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    #masked_original = cv2.putText(masked_original, 'test', (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 10, cv2.LINE_AA)      
    image_path_out = os.path.join(x_test_dir, "output.png")
    cv2.imwrite(image_path_out,masked_original)

    # Load the images from file
    image1 = Image.open(image_input_path)

    image2_path = image_path_out
    image2 = Image.open(image2_path)
    # Resize the images
    width, height = image1.size
    new_size = (int(width/2), int(height/2))
    resized_image1 = image1.resize(new_size)
    resized_image2 = image2.resize(new_size)

    # Display the images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_image1, caption='input')
    with col2:
        st.image(resized_image2, caption='output')

    #converting image co-ordinates to geo location co ordinates
    rect_segments_dict_geo = convert_2_geo_co_ord(rect_segments_dict,long_min,long_max,lat_min,lat_max,width, height)
    #print(type(np.random.randn(50, 5)))
    #print(np.random.randn(50, 5))
    numpy_array = np.zeros((len(rect_segments_dict_geo), 5))
    for i in range(len(rect_segments_dict_geo)):
        numpy_array[i] = [i+1,rect_segments_dict_geo[i+1][0],rect_segments_dict_geo[i+1][1],rect_segments_dict_geo[i+1][2],rect_segments_dict_geo[i+1][3]]


    df = pd.DataFrame(
        numpy_array,
        columns=('roof_index','long_min','long_max','lat_min','lat_max'))

    st.write(f"Total no.of roofs detected -  {len(rect_segments_dict_geo)}")
    st.dataframe(df)


