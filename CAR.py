import os
import random
import re
import numpy as np

class CAR:
    def __init__(self, path,batch_size=10, train_test_split=0.8):
        self.batch_size = batch_size
        self.train_test_split = train_test_split

        self.init_filenames(path)

    def init_filenames(self,path):
        """ Inits filenames. Reads all files contained in subfolders of path.
        These subfolders specify the targetvalues i.e. the classes. Adds an entry to self.files for each file.
        The entry is a dict mapping "sourcefile" to the path and targetval to a string retrieved from the
        foldername (e.g. NIO_Neu_Neu)

        params:
            path : directory of folder with subfolders
        """
        files = []
        self.target_values =[]
        for category_folder in os.listdir(path):
            full_path = path+"/"+category_folder
            description = re.search('TS_(?P<description>.*)_combined.*', category_folder).groups("description")[0]


            files_of_category =[full_path+"/"+filename for filename in os.listdir(full_path)]
            for f in files_of_category:
                files.append({"sourcefile":f,"targetval":description})
            self.target_values.append(category_folder)

        np.random.shuffle(files)
        self.training =files[:int(self.train_test_split*len(files))]
        self.validation = files[int(self.train_test_split*len(files)):]

    def get_training_batches(self):
        self.return_validation = False
        self.return_training = True
        self.current_training_batch = -1
        return iter(self)

    def get_validation_batches(self):
        self.return_validation = True
        self.return_training = False
        self.current_validation_batch = -1
        return iter(self)

    def __iter__(self):
        return self

    def __next__(self):
        if self.return_training:
            if(self.current_training_batch*self.batch_size >= len(self.training)):
                return
            self.current_training_batch += 1

            source_vals = []
            target_vals = []
            for x in range(self.batch_size):
                source_vals.append(np.load(self.training[self.current_training_batch*self.batch_size+x]["sourcefile"]))
                target_vals.append(self.get_target_vector(self.training[self.current_training_batch*self.batch_size+x]["targetval"]))
            return {"source":np.array(source_vals),"target":np.array(target_vals)}

        if self.return_validation:
            if(self.current_validation_batch *self.batch_size >= len(self.training)):
                return
            self.current_validation_batch += 1
            
            source_vals = []
            target_vals = []
            for x in range(self.batch_size):
                source_vals.append(np.load(self.validation[self.current_validation_batch*batch_size+x]["sourcefile"]))
                target_vals.append(self.get_target_vector(self.training[self.current_validation_batch*self.batch_size+x]["targetval"]))
            #target_vals = self.get_target_vector(target_vals)
            return {"source":source_vals,"target":target_vals}

    def get_target_vector(self,description):
        if description == "NIO_Neu_Neu":
            return np.array([0,1,1])

        if description == "Neu_NIO_Neu":
            return np.array([1,0,1])

        if description == "Neu_Neu_NIO":
            return np.array([1,0,1])

        if description == "IO_Neu_Neu":#!!!
            return np.array([1,1,1])

        if description == "Neu_Neu_Neu":
            return np.array([1,1,1])

        if description == "NIO_NIO_Neu":
            return np.array([0,0,1])

        if description == "NIO_NIO_NIO":
            return np.array([0,0,0])

    def assemble_batch(self):
        return



"""car = CAR("/net/projects/scratch/summer/valid_until_31_January_2020/ann4auto/Combined_Chunks/Acc_Vel/1.0s_duration/0.0s_overlap")
myiter = car.get_training_batches()

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))


for x in car.get_training_batches():
    print(".",end="")"""
