import numpy as np
import pandas as pd
import cv2

import redis

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

host_name='redis-17239.c85.us-east-1-2.ec2.cloud.redislabs.com'
port=17239
pwd='ZCojZ6VRsSIqMd3GPu5lJRDKAF1PALNS'

r=redis.StrictRedis(host=host_name,port=port,password=pwd)

faceapp=FaceAnalysis(name='buffalo_sc',root='insight_model',providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

def ml_search_algo(dataframe,feature_column,test_vector,name_role=['Name','Role'],cosine_optimal=0.5):

    data_frame=dataframe.copy()
    x_list=data_frame[feature_column].tolist()
    
    x=np.asarray(x_list)
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    data_frame['cosine']=similar_arr

    data_filter=data_frame.query(f'cosine>={cosine_optimal}')
    if(len(data_filter))>0:
          data_filter.reset_index(drop=True,inplace=True)

          argmax1=data_filter['cosine'].argmax()
    
          name_cos_new,role_cos_new=data_filter.loc[argmax1][['Name','Role']]
    else:
        name_cos_new="unknow"
        role_cos_new="unknow"

    return name_cos_new,role_cos_new




def prediction(test_image,dataframe,feature_column,test_vector,name_role=['Name','Role'],cosine_optimal=0.5):
    results=faceapp.get(test_image)
    test_copy=test_image.copy()
    for res in results:
        x1,y1,x2,y2=res['bbox'].astype(int)
       
        
        embeddings=res['embedding'].reshape(1,-1)
        person_name,role_name=ml_search_algo(dataframe,feature_column,test_vector=embeddings,name_role=name_role,cosine_optimal=0.5)
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),(0,255,0))
        text_gen=person_name
        cv2.putText(test_copy, text_gen, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)  # Fixed 'FONT_HERSEY_DUPLEX' to 'FONT_HERSHEY_DUPLEX'

    return test_copy
    
        
    
    
    