import qctoolkit as qtk
import numpy as np
import os
import datetime
import copy
import glob
import shutil
import time

scratch = '/scratch/samio'
log_file = '/home/samio/remote/maia/algaas_444.db'
root = '/home/samio/02_optimize'
base_root = 'gaas_ref'

mol_base = qtk.Molecule('%s/xyz/gaas_2.xyz' % root)
mol_base.extend([4,4,4])
mol_base.name = base_root

wfk_ref = "%so_DS2_WFK" % base_root
wfk_src = "%s/%s" % (root, wfk_ref)
wfk = "%s/%s" % (scratch, wfk_ref)
wfk_md5 = 'a62f551667214ae820c18190b7f89f51'
den_ref = "%so_DS2_DEN" % base_root
den_src = "%s/%s" % (root, den_ref)
den = "%s/%s" % (scratch, den_ref)
den_md5 = '1895e82dd5a4d7120eeea42f5e27ac94'

ccs = qtk.CCS(mol_base, '%s/ccs.yml' % root)
penalty_input = [ccs, {}]

qmsetting_base = {
    'program': 'abinit',
    'cutoff': 50,
    'kmesh': [3,3,3],
    'band_scan': [
     [20, 20],
     [[1.0, 0.0, 0.5],   # L
      [0.0, 0.0, 0.0],   # Gamma
      [0.0, 1.0, 1.0],   # X
     ]],
    'threads': 12,
    'ks_states': 24,
    'save_restart': True,
    'save_density': True,
    'abinit_setting': ['chkprim 0', 'nsym 1'],
    'unfold': [4,4,4],
}

rst_root = os.path.abspath(base_root)
qmsetting_restart = copy.deepcopy(qmsetting_base)
del qmsetting_restart['threads']
del qmsetting_restart['save_density']
del qmsetting_restart['kmesh']
new_entries = {
    'omp': 3,
    'restart': True,
    'restart_density': True,
    'restart_density_file': den,
    'restart_wavefunction_file': wfk,
    'wf_convergence': 1E5,
    'overwrite': True,
    'link_dep': True,
    'rename_if_exist': True,
    'unfold_cleanup': True,
}
qmsetting_restart.update(new_entries)

def genCCSInp():
    _coord = ccs.random()[1]
    return _coord

def getEg(out, w, fermi = 0.1):
    L, E, W = out.unfold([[0.5,0,0.25], [0,0,0], [0,0.5,0.5]], [4,4,4])
    ind_E = E > fermi
    ind_W = W > w
    ind = ind_E * ind_W
    Eg = min(E[ind])
    ind_cb = np.argmin(E[ind])
    Eg_L_cb = L[ind][ind_cb]

    ind2_E = E <= fermi
    ind2 = ind2_E * ind_W

    ind_vb = np.argmax(E[ind2])
    Eg_L_vb = L[ind2][ind_vb]
    
    return Eg, Eg_L_cb - Eg_L_vb

def get_dep(dep, dep_src, dep_md5):

    dep_lock = dep + '_lock'

    def copy_dep():
        if os.path.exists(dep_lock):
            qtk.warning("copy", "lock file conflict")
        else:
            qtk.progress("copy", dep_src)
            lock_file = open(dep_lock, 'a')
            lock_file.write('copy in progress...')
            lock_file.close()
            shutil.copy(dep_src, dep)
            os.remove(dep_lock)

    def touch_dep():
        with open(dep, 'a'):
            os.utime(dep, None)

    if os.path.exists(dep):
        # if copy is in progress
        while os.path.exists(dep_lock):
            time.sleep(5)
        md5 = qtk.md5sum(dep)
        if md5 == dep_md5:
            # everything is good!
            touch_dep()
        else:
            # something wrong! copy finished by file corrupted 
            copy_dep()
    else:
        copy_dep()
            
def penalty_function(ccs_coord, ccs, qmsetting_dict={}):
    # heavy computation here!

    get_dep(den, den_src, den_md5)
    get_dep(wfk, wfk_src, wfk_md5)
    
    qmsetting = copy.deepcopy(qmsetting_restart)
    qmsetting.update(qmsetting_dict)
    
    node_name = os.uname()[1].replace('.cluster.bc2.ch', '')
    time_stamp = datetime.datetime.now()
    
    mol_mut = ccs.generate(**ccs_coord)
    mol_mut.name = '%s_%s' % (node_name, time_stamp.strftime('%m%d%H%M%S%f')[:10])
    mol_mut.name = mol_mut.name + '_' + str(os.getpid())[-3:]
    inp_mut = qtk.QMInp(mol_mut, **qmsetting)

    out_mut = inp_mut.run()
    
    Eg, k = getEg(out_mut, 0.5)
    
    if abs(k) < 1E-2:
        score = Eg
    else:
        score = 0
    
    return score, mol_mut.name

optimzer = qtk.optimization.GeneticOptimizer(
    penalty_function, 
    penalty_input, 
    genCCSInp, 
    ccs.mate, 
    20, 
    target=5, 
    threads=1,
    mode='maximize',
    new_run=False,
    log=log_file,
    max_step=1,
)

optimzer.run()
