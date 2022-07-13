import os

list= ['1a', '2a', '3a', '4a', '1b', '2b', '3b', '4b', '1c', '2c', '3c', '4c']
working_path='/Volumes/scratch/neurobiology/zimmer/Tanja/slides_20220712/'

for element in list:
    #folder_name=os.path.join(working_path,genotype+str(i))
    print(type(element))
    folder_name='odr-10_stacks_line_'+element
    print(folder_name)
    folder_path=os.path.join(working_path, folder_name)
    print(folder_path)
    os.mkdir(folder_name)