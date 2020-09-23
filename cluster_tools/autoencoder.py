import numpy as np
from keras.models import load_model
import sys
import os.path
sys.path.insert(0, '/Users/mishamesarcik/Workspace/phd/Workspace/lofar-dev/')
import preprocessor
from numpy import sqrt,angle


class Autoencoder(object):
    def __init__(self,encoder_path,config):
        self.model= load_model(encoder_path,compile=False)
        self.config = config
        self.encoder = self.model.get_layer('encoder') 
        self.incompatible= False 

    def encode(self, complex_db_cube):
        """
        Encodes the cube with a pretrained encoding Keras model that is specified in settings.
        :param complex_db_cube: numpy.array with shape (baseline,subband,timestamp,pol
        :return: numpy.array with shape (baseline,D) where D is dependent on input shape size and model.
        """
        if complex_db_cube is None or len(complex_db_cube.shape) != 4:
            raise ValueError('Data is not in correct format: numpy.array with shape (baseline,subband,timestamp,pol)')
                                                                
        print('This is complex cube shape after mean {}'.format(complex_db_cube.shape))
        ##################################3
        complex_db_cube= np.concatenate([complex_db_cube[...,0:1],
                                  complex_db_cube[...,4:5]],axis=3)
        print('This is complex cube shape after mean {}'.format(complex_db_cube.shape))
        ##################################3
        p = preprocessor.preprocessor(complex_db_cube)
        p.interp(self.config['n_frequencies']['value'], self.config['n_time_steps']['value'])
        cube = p.get_processed_cube()
         
        if self.config['architecture']['value'] =='skip':
            p.get_phase()
            phase_cube = p.get_processed_cube()
            p = preprocessor.preprocessor(cube)
            p.get_magnitude()
            p.median_threshold()
            p.minmax(per_baseline=True,feature_range=(np.min(phase_cube),np.max(phase_cube)))
            encoded,_,_ = self.encoder.predict([p.get_processed_cube(),phase_cube])
            return encoded.reshape(encoded.shape[0], np.product(encoded.shape[1:]))


        elif self.config['mag_phase']['value']:
            p.get_magnitude_and_phase() 

        elif self.config['magnitude']['value']:
            p.get_magnitude() 
            p_cube = p.get_processed_cube()
            self.config['n_layers']['value'] = p_cube.shape[-1] # TODO: This might cause problems 
        elif self.config['phase']['value']:
            p.get_phase() 
            p_cube = p.get_processed_cube()
            self.config['n_layers']['value'] = p_cube.shape[-1] # TODO: This might cause problems 

        if self.config['median_threshold']['value']:
            p.median_threshold(per_baseline=self.config['per_baseline']['value'])

        if self.config['db']['value']:
             p.mag2db()

        if self.config['wavelets']['value']:
            p.wavelet_decomp_2D()


        if self.config['flag']['value']:
            #TODO
             raise Exception('Flagging Code Not Written')
        if self.config['freq']['value']:
            #TODO
             raise Exception('Frequency Domain Code Not Written')
        if self.config['standardise']['value']:
            p.standardise(per_baseline=self.config['per_baseline']['value'])

        elif self.config['minmax']['value']:
            p.minmax(per_baseline=self.config['per_baseline']['value'])



        real_cube = p.get_processed_cube()

        #use preprocessor to reshape cubes
        if self.config['architecture']['value'] == 'vae':
            encoded,_,_ = self.encoder.predict(real_cube)
        else: encoded = self.encoder.predict(real_cube)
        #encoded  = np.mean(real_cube,axis=3)[:,::4,::4]
        print('This is complex cube shape after mean {}'.format(encoded.shape))
        return encoded.reshape(encoded.shape[0], np.product(encoded.shape[1:]))

    def wavelet_decomp(self,data):
        p = preprocessor.preprocessor(data)
        p.wavelet_decomp_2D()
        return p.get_processed_cube()
