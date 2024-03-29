import os
import sys

files = []
def rek_get_files(path, name_contains):
    for f in os.scandir(path):
        if f.is_dir():
            rek_get_files(f.path, name_contains)
        else:
            if name_contains in f.name:
                files.append(path+"/"+f.name)

def get_files(path, name_contains):
    files.clear()
    rek_get_files(path, name_contains)
    return files


def submit_script(scriptpath, args, mem = "5G", cuda_cores = 0,jobname = "job"):
    args_string = ""
    for a in args:
        args_string += a
        args_string += " "
    args_string = args_string[:-1]


    os.system('qsub -l mem='+ mem +' -l cuda_cores='+ str(cuda_cores)+' <<eof\n'
                + "#!/bin/bash \n"
                + "#$ -N " + jobname +"\n"
                + "#$ -l h_rt=01:30:00" + "\n"
                + "echo 'Start-time' \n"
                + "date \n"
                + "echo 'Host' \n"
                + "hostname \n"
                + 'source /net/projects/scratch/summer/valid_until_31_January_2020/ann4auto/env/gpuenv/bin/activate \n'
                + 'python3 ' + scriptpath + " " + args_string + "\n"
                + "echo 'End-time'\n"
                + "date \n"
                + '\n'+'eof')
    
if __name__ == "__main__":
    submit_script(os.getcwd()+"/"+sys.argv[1],sys.argv[2:])

