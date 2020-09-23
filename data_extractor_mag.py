import aipy, uvtools
import numpy as np
import pylab as plt
from hera_sim import foregrounds, noise, sigchain, rfi
import datetime
import random
import pickle
from tqdm import tqdm
import pandas as pd
from preprocessor import preprocessor

fqs = np.linspace(.1,.2,64,endpoint=False)
lsts = np.linspace(0,2*np.pi,256, endpoint=False)
times = lsts / (2*np.pi) * aipy.const.sidereal_day
bl_len_ns_list = [1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
MX, DRNG = 2.5, 3
Tsky_mdl = noise.HERA_Tsky_mdl['xx']
g = sigchain.gen_gains(fqs, [1,2,3])

class HERA_generator():
    def __init__(self):
        self.bl_len_ns = bl_len_ns_list[random.randint(0,len(bl_len_ns_list)-1)]
        self.nsrcs = random.randint(1,1001) 
    
    def generate_diffuse_foreground(self):
        return foregrounds.diffuse_foreground(lsts, fqs, self.bl_len_ns,Tsky_mdl=Tsky_mdl)/40

    def generate_point_source(self):
        return foregrounds.pntsrc_foreground(lsts, fqs, self.bl_len_ns, nsrcs=self.nsrcs) 
        
    def generate_noise(self):
        omega_p = noise.bm_poly_to_omega_p(fqs)
        tsky = noise.resample_Tsky(fqs,lsts,Tsky_mdl=noise.HERA_Tsky_mdl['xx'])
        t_rx = 150.
        return noise.sky_noise_jy(tsky + t_rx, fqs, lsts,omega_p)

    def generate_rfi_stations(self):
        return  rfi.rfi_stations(fqs, lsts)/200
    
    def generate_rfi_impulse(self):
        return rfi.rfi_impulse(fqs, lsts,strength=300, chance=.05)
    
    def generate_rfi_scatter(self):        
        return rfi.rfi_scatter(fqs, lsts, strength=600,chance=.01)
    
    def generate_rfi_dtv(self):
        return rfi.rfi_dtv(fqs, lsts, strength=500,chance=.1)

    def generate_gains(self,vis):
        gainscale = np.average([np.median(np.abs(g[i])) for i in g])
        MXG = MX + np.log10(gainscale)
        return sigchain.apply_gains(vis, g, (1,2))

    def generate_x_talk(self,vis):
        xtalk = sigchain.gen_cross_coupling_xtalk(fqs,np.ones(fqs.shape))
        return sigchain.apply_xtalk(vis, xtalk)

    functions = [generate_rfi_scatter,
                 generate_rfi_impulse,
                 generate_rfi_stations,
                 generate_rfi_dtv,
                 generate_noise,
                 generate_point_source,
                 generate_diffuse_foreground]
                

def cointoss_addition(i,o):
    c = 1
    if c:
        o +=i
    return o,c,i

def generate_mask(signal):
    mask = np.zeros([32,128,1])
    for i in range(32):
        for j in range(128):
            if signal[i,j]> 2:
                mask[i,j] = 1
    return mask

def generate_mask1(signal):
    mask = np.zeros([32,128,1])
    for i in range(32):
        for j in range(128):
            if (signal[i,j] > 15) or  (signal[i,j] > 25) or (signal[i,j] > 35):
                mask[i,j] = 1
    return mask

def generate_mask2(signal):
    mask = np.zeros([32,128,1])
    for i in range(32):
        for j in range(128):
            if signal[i,j]> 0.5:
                mask[i,j] = 1
    return mask

def generate_mask3(signal):
    mask = np.zeros([32,128,1])
    for i in range(32):
        for j in range(128):    
            if signal[i,j]> 60:
                mask[i,j] = 1
    return mask

def generate_label(labels):
    label_str = ''
    for key in labels:
        if labels[key] != 0:
            label_str = label_str + '-' + key
    return label_str[1:]

def to_magnitude(complex_data):
    data = np.array([complex_data.real,complex_data.imag])
    data = np.swapaxes(np.array(data),0,-1)
    p = preprocessor(np.expand_dims(data,axis=0))
    p.interp(32,128) # interpolate 
    p.get_magnitude()
    return p.get_processed_cube()[0,...] # dont want the first component as this comes when we concatenate


def generate_vis(mag=False):
    hera = HERA_generator()
    labels = {'noise':1,
              'point_source':0,
              'rfi_stations':0,
              'rfi_impulse':0,
              'rfi_dtv':0,
              'gains':0,
              'x_talk':0
              }
    if mag:
        output_vis = to_magnitude(hera.generate_noise())
        output_vis,label,sig1 = cointoss_addition(to_magnitude(hera.generate_point_source()),output_vis)
        mask1 = generate_mask1(sig1)
        labels['point_source'] = label 

        output_vis,label,sig2 = cointoss_addition(to_magnitude(hera.generate_rfi_impulse()),output_vis)
        mask2 = generate_mask2(sig2)
        labels['rfi_impulse'] = label

        output_vis,label,sig3 = cointoss_addition(to_magnitude(hera.generate_rfi_stations()),output_vis)
        mask3 = generate_mask3(sig3)
        labels['rfi_stations'] = label
        
        output_vis,label,sig4 = cointoss_addition(to_magnitude(hera.generate_rfi_dtv()),output_vis)
        mask4 = generate_mask(sig4)
        labels['rfi_dtv'] = label

    else:
        output_vis = hera.generate_noise() 
        output_vis,label = cointoss_addition(hera.generate_point_source(),output_vis)
        labels['point_source'] = label 

        output_vis,label = cointoss_addition(hera.generate_rfi_impulse(),output_vis)
        labels['rfi_impulse'] = label

        output_vis,label = cointoss_addition(hera.generate_rfi_stations(),output_vis)
        labels['rfi_stations'] = label
        
        output_vis,label = cointoss_addition(hera.generate_rfi_dtv(),output_vis)
        labels['rfi_dtv'] = label

        if random.randint(0,1):
            output_vis = hera.generate_gains(output_vis)
            labels['gains'] =1 
        if random.randint(0,1):
            output_vis  = hera.generate_x_talk(output_vis) 
            labels['x_talk'] =1 

    return output_vis,generate_label(labels),mask1,mask2,mask3,mask4

def make_complex_dataset(n):
    data,label = [],[]

    for i in  tqdm(range(0,n)):
        g,l = generate_vis()
        data.append(np.array([g.real,g.imag]))
        label.append(l)
    
    return np.swapaxes(np.array(data),1,3),np.array(label)

def make_mag_dataset(n):
    data,label,mask1,mask2,mask3,mask4 = [],[],[],[],[],[]

    for i in  tqdm(range(0,n)):
        g,l,m1,m2,m3,m4 = generate_vis(mag=True)
        data.append(g)
        label.append(l)
        mask1.append(m1)
        mask2.append(m2)
        mask3.append(m3)
        mask4.append(m4)

    
    return np.array(data),np.array(label),np.array(mask1),np.array(mask2),np.array(mask3),np.array(mask4)

def main():
    data,labels,mask1,mask2,mask3,mask4 = make_mag_dataset(6250)
    info = {'Description':'Hera training set with only magnitude component',
            'Features':pd.unique(labels),
            'Dimensions':(32,128),
            'Source':'HERA Simulator'}

    pickle.dump([data,np.zeros([1,1,1,1]),labels, np.zeros([1,1,1,1]),mask1,mask2,mask3,mask4,info],
            open('/var/scratch/nsc400/hera_data/HERA5k_masks{}.pkl'.format(datetime.datetime.now().strftime("%d-%m-%Y")), 'wb'), protocol=4)
    

if __name__ =='__main__':
    main()


