import os
from tqdm import tqdm
import re
from util.vtk_revise import read_vtk
from get_thickness_with_interpolate import get_thick_norm_and_ori



rootDir = './AllCortexData'


for dirName, subdirList, fileList in os.walk(rootDir):
    
    
    
    read_dir_name = dirName.replace("\\", "/")
    if 'SUBJ' in dirName : 
        subject_num = re.findall(r'\d+', read_dir_name)[0]
        
    full_file_name_lh_white = None
    full_file_name_lh_pial = None
    full_file_name_rh_white = None
    full_file_name_rh_pial  = None
    
    for fname in fileList:
        if 'vtk' in fname:
            if 'lh' in fname : 
                if 'white' in fname : 
                    full_file_name_lh_white = read_dir_name + '/' + fname
                    
                elif 'pial' in fname : 
                    full_file_name_lh_pial = read_dir_name + '/' + fname
                    
                
                
            elif 'rh' in fname : 
                if 'white' in fname : 
                    full_file_name_rh_white = read_dir_name + '/' + fname
                elif 'pial' in fname : 
                    full_file_name_rh_pial = read_dir_name + '/' + fname

    
    if full_file_name_lh_white != None and \
        full_file_name_lh_pial != None and \
        full_file_name_rh_white != None and \
        full_file_name_rh_pial != None : 
            
        ### lh brain ###
        print(f"-----------------SUBJECT {subject_num} LH BRAIN START----------------")
        lh_white = read_vtk(full_file_name_lh_white) 
        lh_pial = read_vtk(full_file_name_lh_pial)  
        
        if 'BL' in full_file_name_lh_white :
            lh_file_name = 'SUBJ_' +  subject_num + '_lh_white_BL'

        elif 'FU' in full_file_name_lh_white :
            lh_file_name = 'SUBJ_' +  subject_num + '_lh_white_FU'
        
        get_thick_norm_and_ori(lh_file_name,lh_white,lh_pial,50)
        
        



        ### rh brain ###
        print(f"-----------------SUBJECT {subject_num} RH BRAIN START----------------")
        rh_white = read_vtk(full_file_name_rh_white)  
        rh_pial = read_vtk(full_file_name_rh_pial)  

        if 'BL' in full_file_name_rh_white :
            rh_file_name = 'SUBJ_' +  subject_num + '_rh_white_BL'

        elif 'FU' in full_file_name_rh_white :
            rh_file_name = 'SUBJ_' +  subject_num + '_rh_white_FU'

        get_thick_norm_and_ori(rh_file_name,rh_white,rh_pial,50)



               

print('#####------- ALL PROSCESS DONE -------#####')
