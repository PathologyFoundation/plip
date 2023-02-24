
import sys, os, platform, copy
import pandas as pd
import numpy as np
import itertools
from collections import Counter
from tqdm import tqdm
opj = os.path.join


if __name__ == '__main__':

    override = True

    partition = 'jamesz'
    #partition = 'gpu'

    train_dataset = 'All'
    train_dataset = 'TRL' # TRL, TR, L

    if partition == 'jamesz':
        n = 8
    elif partition == 'owners':
        n = 20
    elif partition == 'gpu':
        n = 8

    wd = '/oak/stanford/groups/jamesz/pathtweets/ML_scripts/part_5_linear_probing/path_eval'
    bash_folder = opj(wd, 'parallel_on_sherlock', 'bash_%s_%s' % (partition, train_dataset))
    os.makedirs(bash_folder, exist_ok=True)

    resultdir = '/oak/stanford/groups/jamesz/pathtweets/v2/plip_results_linear_2'
    os.makedirs(resultdir, exist_ok=True)

    datasets = [
                'Kather',
                #'WSSS4LUAD_binary',
                #'DigestPath',
                #'PanNuke',
                #'BreaKHis_benign_malignant',
                #'GasHisSDB',
                #'BreaKHis_8_subtypes',
                ]
    alphas = [0.1, 0.01, 0.001, 0.0001]

    models = []
    model_rootdir = '/oak/stanford/groups/jamesz/pathtweets/v2/plip_models'
    for m in os.listdir(model_rootdir):
        fp = opj(model_rootdir, m)
        if fp.endswith('.pt'):
            if train_dataset != 'All':
                this_train_data = fp.split("comet_tags_['")[1].split('.csv')[0]
                if this_train_data == train_dataset:
                    models += [fp]
                else:
                    pass
    
    baseline_model_list = ['clip', 'mudipath']
    models = models + baseline_model_list

    #print('-----------------------------------------------------')
    #print('List all available models (%d):' % len(models))
    #for m in models: print(m)
    #print('-----------------------------------------------------')

    
    if override:
        print('remove all files in bash_folder')
        filelist = [f for f in os.listdir(bash_folder)]
        for f in filelist: os.remove(os.path.join(bash_folder, f))

    all_combinations = itertools.product(
                                        models,
                                        datasets,
                                        alphas,
                                        )


    text_list = []

    n_finished = 0
    n_total = 0
    for combination in all_combinations:
        modelfullpath, dataset, alpha = combination
        n_total += 1
        
        if modelfullpath in baseline_model_list:
            modelname = modelfullpath
            backbone = 'default'
        else:
            modelname = 'plip'
            backbone = modelfullpath # the backbone should be full path

        os.makedirs(opj(resultdir, dataset, modelname, str(alpha)), exist_ok=True)
        result_fullpath = opj(resultdir, dataset, modelname, str(alpha), '%s.csv' % backbone)
        print(result_fullpath)
        if os.path.exists(result_fullpath):
            n_finished += 1
            #os.remove(result_fullpath)
            continue
        bashtext = 'python linear_probing_evaluation.py' + \
                    ' --backbone "%s"' % str(modelfullpath) + \
                    ' --dataset "%s"' % str(dataset) + \
                    ' --alpha "%s"' % str(alpha)
        text_list.append(bashtext)
    

    print('-----------------------------------------------------')
    print('Finished %d/%d, now working on %d combinations.' % ( n_finished, n_total, n_total-n_finished ) )
    print('-----------------------------------------------------')
    
    if not override:
        sys.exit()
    n_text_list = np.array_split(np.array(text_list), n)

    for i, tl in enumerate(n_text_list):
        bashtext = "conda activate pathCLIP \n"
        bashtext += "ml load py-numpy/1.20.3_py39 \n"
        bashtext += "cd %s \n" % os.path.join(wd, 'scripts')
        
        if partition == 'gpu':
            tl = np.array(tl)[::-1]



        job_list = ''
        for t in tl:
            job_list += t + ';\n'
        bashtext = bashtext + job_list
        with open(os.path.join(bash_folder, 'run_bash_%d.sh' % (i+1)), 'w') as f:
            f.write(bashtext)
        # np.savetxt(os.path.join(bash_folder, 'run_bash_%d.sh' % (i+1)), tl, fmt = "%s")
        # https://stackoverflow.com/questions/57044117/error-only-when-submitting-python-job-in-slurm
        '''
        To use the in-group GPU, you should use this flag:
        -p jamesz
        Otherwise, use:
        -p gpu
        '''
        if partition == 'gpu':
            gpu_config = "#SBATCH -p gpu \n" + \
                        "#SBATCH -C GPU_BRD:GEFORCE \n" + \
                        "#SBATCH -C GPU_MEM:16GB \n"
        elif partition == 'jamesz':
            #gpu_config = "#SBATCH -p jamesz \n" + \
            #             '#SBATCH -C "GPU_SKU:RTX_2080Ti" \n'
            gpu_config = "#SBATCH -p jamesz \n"
        elif partition == 'owners':
            gpu_config = "#SBATCH -p owners \n"


        jobscript = "#!/bin/bash \n" + \
                    gpu_config + \
                    "#SBATCH --gpus-per-task=1 \n" + \
                    "#SBATCH --gpus-per-node=1 \n" + \
                    "#SBATCH -J L%d_%s \n" % (i+1, train_dataset) + \
                    "#SBATCH -o L%d_%s.txt\n" % (i+1, train_dataset) + \
                    "#SBATCH -e L%d_%s.err\n" % (i+1, train_dataset) + \
                    "#SBATCH --time=47:59:00 \n" + \
                    "#SBATCH --ntasks=1 \n" + \
                    "#SBATCH --cpus-per-task=1 \n" + \
                    "#SBATCH --mem-per-cpu=16G \n" + \
                    "bash %s/run_bash_%d.sh" % (bash_folder ,i+1)
                    
        with open(os.path.join(bash_folder, 'job_%d.script' % (i+1)), 'w+') as f:
            f.write(jobscript)


    qsub = []
    for i in range(n):
        qsub.append('sbatch job_%d.script' % (i+1))
    
    np.savetxt(os.path.join(bash_folder, '_sbatch_all.sh'), qsub, fmt = "%s")

    print('----------------------------')
    print('Don\'t forget:')
    print('conda activate pathCLIP')
    print('----------------------------')




