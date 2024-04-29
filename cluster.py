'''author: Wenchang Yang (yang.wenchang@uci.edu)'''

import xarray as xr
from sklearn.cluster import KMeans as KMeans_sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class KMeans(KMeans_sklearn):
    '''Inherit from sklearn.cluster.KMeans that accept xarray.DataArray data
    to fit.'''

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        
    def __str__(self):
        attrs = ['algorithm', 'copy_x', 'init', 'max_iter',
            'n_clusters', 'n_init', 
            'random_state', 'tol', 'verbose']
        #ini dihapus karena tidak ada jadi erorr: 'n_jobs', 'precompute_distances',
        # s = ', '.join(['{}={}'.format(a, getattr(self, a)) for a in attrs])
        values = []
        for attr in attrs:
            #print(attr)
            value = getattr(self, attr)
            #print(value)
            if isinstance(value, str):
                value = '"{}"'.format(value)
            values.append(value)
        #print(values)
        s = ', '.join(['{}={}'.format(attr, value)
            for attr,value in zip(attrs, values)])
        return '[KMeans for xarray]: {}.'.format(s)
        
    def __repr__(self):
        return self.__str__()

    def fit(self, da):
        print('tes tes')
        '''xarray.DataArray version of sklearn.cluster.KMeans.fit.'''

        # Compatible with the slearn.cluster.KMeans fit method when the input data is not DataArray
        if not isinstance(da, xr.DataArray):
            return super().fit(da)

        # retrieve parameters
        n_samples = da.shape[0] #0 time=12 annual_cycle => feature dan z(x,y)=> samples
        samples_dim = da.dims[0] #0 time
        samples_coord = {samples_dim: da.coords[samples_dim]}
        
        features_shape = da.shape[1:] #lat= lon=
        n_features = np.prod(features_shape) #lat*lon
        print('n_features', n_features)
        
        features_dims = da.dims[1:] #(lat,lon)
        cluster_centers_dims = ('cluster',) + features_dims  #(cluster,lat,lon)
        cluster_centers_coords = {'cluster': np.arange(self.n_clusters)}
        for dim in features_dims:
            print(dim)
            cluster_centers_coords[dim] = da.coords[dim]

        # transform the input data array
        print('da.shape', da.shape)
        X = da.data.reshape(n_samples, n_features) 
        print(X)
        print('X.shape', X.shape)
        #'data' might be replaced with 'values'
        
        # any feature contains np.NaN
        valid_features_index = ~np.isnan(X[0,:])
        X_valid = X[:, valid_features_index]
      
        print('valid_features_index',valid_features_index)
        print('valid_features_index.shape',valid_features_index.shape)
        print('X_valid',X_valid)
        print('X_valid.shape', X_valid.shape)
        '''
        valid_features_index = ~np.isnan(X_valid[:,0])
        X_valid = X_valid[valid_features_index,:]
        print('X_valid.shape', X_valid.shape)
        '''
        super().fit(X_valid) # call 
        print(self)
        print('self.cluster_centers_.shape',self.cluster_centers_.shape)
        print('self.labels_', self.labels_)
        print('self.labels_.shape', self.labels_.shape)
        
        
        self.valid_features_index_ = valid_features_index

        # wrap the estimated parameters into DataArray
        cluster_centers = np.zeros((self.n_clusters, n_features)) + np.NaN
        
        cluster_centers[:, valid_features_index] = self.cluster_centers_
        
        print('cluster_centers.shape', cluster_centers.shape)
        
        
        self.cluster_centers_da = xr.DataArray(
            cluster_centers.reshape((self.n_clusters,) + features_shape),
            dims=cluster_centers_dims,
            coords=cluster_centers_coords
        )
        self.labels_da = xr.DataArray(self.labels_, dims=samples_dim,
            coords=samples_coord)
        print('self.cluster_centers_da', self.cluster_centers_da)
        return self
    
    def fit2(self, da, annual_cycle=False, with_pca=False):
        '''xarray.DataArray version of sklearn.cluster.KMeans.fit.'''
        print('-------- clustering... fit2... ------------------')
        # Compatible with the slearn.cluster.KMeans fit method when the input data is not DataArray
        if not isinstance(da, xr.DataArray):
            return super().fit(da)
        #da=da.fillna(0)
        #da=da.dropna(dim='month', how='any', thresh=None)
        # retrieve parameters
        n_samples = da.shape[0] #0 time=12
        samples_dim = da.dims[0] #0 time
        samples_coord = {samples_dim: da.coords[samples_dim]}
        
        features_shape = da.shape[1:] #lat= lon=
        n_features = np.prod(features_shape) #lat*lon
        #print('n_features', n_features)
        
        features_dims = da.dims[1:] #(lat,lon)
        cluster_centers_dims = ('cluster',) + features_dims  #(...,lat,lon)
        cluster_centers_coords = {'cluster': np.arange(1)}
        for dim in features_dims:
            #print(dim)
            cluster_centers_coords[dim] = da.coords[dim]
        
        #print('da.shape', da.shape)
        #------ transform the input data array
        #X = da.data.reshape(n_features, n_samples) # ini dibalik
        X=da.T
        print('dimensi input_tranfose=', X.shape)
        #print(X)
        #print(X.shape)
        #'data' might be replaced with 'values'
        
        # any feature contains no np.NaN 
        # dim X[z,time] dicek hanya pada t=0
        valid_features_index = ~np.isnan(X[:,0]) #ini
        #print('valid_features_index',valid_features_index)
        #print('valid_features_index',valid_features_index.data)
        #print('valid_features_index.shape',valid_features_index.shape)
        
        X_valid = X[valid_features_index.data,:]  # ini juga
                                             # error ada nan
        
        
        #valid_features_index = ~np.isnan(X_valid[0,:])
        #X_valid = X_valid[:,valid_features_index.data]
        
        #X_valid.fillna(0)
        #X_valid[np.isnan(X_valid)] = 0

        #x = x[~numpy.isnan(x)]
        #print('X_valid.shape', X_valid.shape)
        #print('X_valid', X_valid)
        
       
        #scaler = StandardScaler()
        #X_valid = scaler.fit_transform(X_valid)
        
        print('dimensi input_valid=', X_valid.shape)
        print('X_valid.shape[0]',X_valid.shape[0])
        print('X_valid.shape[1]',X_valid.shape[1])
        
         #monthly ada nan nya
        if not annual_cycle and not with_pca:
            X_valid=X_valid.fillna(0)
            #print('X_valid=', X_valid)
            '''
            for i in np.arange(X_valid.shape[0]):
                #for j in np.arange(X_valid.shape[1]):
                print(i)
                print('np.isnan(X_valid)=',X_valid[i,:].values)
            #untuk monthly data ini ada nan dan akan error saat fit()
            '''
        
        #jika ini, month perlu diganti ke time, jika tidak maka error
        if annual_cycle: 
            X_valid = X_valid.rename({'month':'time'})
        #print('X_valid', X_valid)
       
        #PCA ini mengurangi jumlah var/features/dimensi data
        #Bisa 'juga' untuk memilih samples dominan namun 
        #tidak/belum bisa memberi label seperti K-means 
        

        #import matplotlib.pyplot as plt
        #fig=plt.figure()
        if with_pca:
            from eofs2.xarray import Eof
            solver = Eof(X_valid)
            pc = solver.pcs2(npcs=3, pcscaling=0)
            print('dimensi hasil_pca=', pc.shape)
            X_valid =pc
            print('dimensi hasil_pra_kmeans=', X_valid.shape)
            print(X_valid)
            
            # components that explain ~90% of the cumulative variance
            cumulative_variance = np.cumsum(solver.varianceFraction()).data
            print('cumulative_variance', cumulative_variance)
            
            cut_off_point = np.argmax(cumulative_variance >= 0.90) + 1
            
            print(f"Number of principal components explaining ~90% of cumulative variance: {cut_off_point}")
            
            # Plot explained variance
            plt.plot(range(1, len(solver.varianceFraction()) + 1), cumulative_variance, marker='o')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Variance Explained')
            plt.title('Scree Plot')
            plt.show()
            
            
            '''
            for i in np.arange(X_valid.shape[0]):
                if i<10:
                    plt.plot(np.arange(X_valid.shape[1])+1,X_valid[i], label=i)
            plt.legend()
            plt.title('PCA in z (z,T) dengan T annual cycle 12 month')
            plt.show()
            
            #exit()
            '''
        #X_valid=X_valid.fillna(0)
        super().fit(X_valid) # call 
        print('dimensi hasil_kmeans=', X_valid.shape)
        print(self)
        #print('self.cluster_centers_.shape',self.cluster_centers_.shape)
        #print('self.labels_', self.labels_)
        #print('self.labels_min', self.labels_.min())
        #print('self.labels_.shape', self.labels_.shape)
        
        #-----pada tahap tsb self sudah berisi center dan label
        #-----center bisa di plot line, versi map gak bisa 
        
        #-----Then buat xarray untuk plot hasil
        self.valid_features_index_ = valid_features_index
        
        # wrap the estimated parameters into DataArray
        #cluster_centers = np.zeros((self.n_clusters, n_features)) + np.NaN
        cluster_centers = np.zeros((n_features,1)) + np.NaN
        print('cluster_centers.shape', cluster_centers.shape)
        
        #cluster_centers[:, valid_features_index] = self.cluster_centers_
        cluster_centers[valid_features_index.data,0] = self.labels_
        #print(cluster_centers)
        
        self.cluster_centers_da = xr.DataArray(
            cluster_centers.reshape((1,) + features_shape),
            dims=cluster_centers_dims,
            coords=cluster_centers_coords
        )
        self.cluster_centers_da=self.cluster_centers_da.unstack()
        print('dimensi hasil_final=', X_valid.shape)
        '''
        #self.labels_da = xr.DataArray(self.labels_, dims=samples_dim,
        #    coords=samples_coord)
        
        
        #print('self.cluster_centers_da', self.cluster_centers_da)
        #print('self.cluster_centers_da_min_max', 
        #        self.cluster_centers_da.min().data, 
        #        self.cluster_centers_da.max().data)
        #print('=====================================')
        '''
        return self
    
    def fit3(self, da, annual_cycle=False, with_pca=False):
        '''xarray.DataArray version of sklearn.cluster.KMeans.fit.'''
        print('-------- clustering... fit3 .... ------------------')
        # Compatible with the slearn.cluster.KMeans fit method when the input data is not DataArray
        if not isinstance(da, xr.DataArray):
            return super().fit(da)
        #da=da.fillna(0)
        #da=da.dropna(dim='month', how='any', thresh=None)
        # retrieve parameters
        n_samples = da.shape[0] #0 time=12
        samples_dim = da.dims[0] #0 time
        samples_coord = {samples_dim: da.coords[samples_dim]}
        
        features_shape = da.shape[1:] #lat= lon=
        n_features = np.prod(features_shape) #lat*lon
        #print('n_features', n_features)
        
        features_dims = da.dims[1:] #(lat,lon)
        cluster_centers_dims = ('cluster',) + features_dims  #(...,lat,lon)
        cluster_centers_coords = {'cluster': np.arange(1)}
        for dim in features_dims:
            #print(dim)
            cluster_centers_coords[dim] = da.coords[dim]
        
        #print('da.shape', da.shape)
        # transform the input data array
        #X = da.data.reshape(n_features, n_samples) # ini dibalik
        X=da.T
        #print('X.shape', X.shape)
        # any feature contains np.NaN
        valid_features_index = ~np.isnan(X[:,0]) #ini
        X_valid = X[valid_features_index.data,:]  # ini juga
        #print('X_valid.shape', X_valid.shape)
        #print('X_valid', X_valid)
        
        '''
        #for raw monthly => error contain NaN
        # ini tetap error=> pada grid tidak nan namun pada time ada nan
        for i in np.arange(X_valid.shape[0]):
            valid_features_index = ~np.isnan(X_valid[i,:]) #ini
            #print('valid_features_index', valid_features_index)
            print('X_valid[i,:]',X_valid[i,:])
        X_valid = X_valid[i,valid_features_index.data]
        
        
        print('X_valid.shape', X_valid.shape)
        '''
        #monthly ada nan nya
        if not annual_cycle and not with_pca:
            X_valid=X_valid.fillna(0)
        
        if annual_cycle:
            X_valid = X_valid.rename({'month':'time'})
            
        if with_pca:
            from eofs2.xarray import Eof
            solver = Eof(X_valid)
            pc = solver.pcs2(npcs=12, pcscaling=0)
            print('dimensi hasil_pca=', pc.shape)
            X_valid =pc
            print('dimensi hasil_pra_kmeans=', X_valid.shape)
            #print(pc)
        
        super().fit(X_valid) # call 
        #for raw monthly => error contain NaN
        
        
        self.valid_features_index_ = valid_features_index

        # wrap the estimated parameters into DataArray
        cluster_centers = np.zeros((n_features,1)) + np.NaN
        #print('cluster_centers.shape', cluster_centers.shape)
        
        
        cluster_centers[valid_features_index.data,0] = self.labels_
        #print(cluster_centers)
        
        self.cluster_centers_da = xr.DataArray(
            cluster_centers.reshape((1,) + features_shape),
            dims=cluster_centers_dims,
            coords=cluster_centers_coords
        )
        self.cluster_centers_da=self.cluster_centers_da.unstack()
        
        
        #self.labels_da = xr.DataArray(self.labels_, dims='label',
        #    coords=samples_coord)
       
        #print('X_valid',X_valid)
        #print('X_valid.s',X_valid.shape)
        #print('self.labels_', self.labels_)
        #print('self.labels_.s', self.labels_.shape)
        
        return self, X_valid


    def predict(self, da):
        '''xarray.DataArray version of sklearn.cluster.KMeans.fit.'''

        # compatible with the sklean.cluster.KMeans predict method when the input data is not DataArray
        if not isinstance(da, xr.DataArray):
            return super().predict(da)

        # retrieve parameters
        n_samples = da.shape[0]
        features_shape = da.shape[1:]
        n_features = np.prod(features_shape)

        X = da.data.reshape(n_samples, n_features)# 'data' might be replaced with 'values'.
        # remove NaN values if exists in X
        try:
            X_valid = X[:, self.valid_features_index_]
        except:
            X_valid = X
        samples_dim = da.dims[0]
        samples_coord = {samples_dim: da.coords[samples_dim]}
        labels = xr.DataArray(super().predict(X_valid),
            dims=samples_dim, coords=samples_coord)

        return labels

    def get_cluster_fraction(self, label):
        '''Get the fraction of a given cluster denoted by label.'''
        return (self.labels_==label).sum()/self.labels_.size

    def plot_cluster_centers(self, label=None, **kw):
        '''Plot maps of cluster centers.'''
        if label is None:
            label = range(self.n_clusters)
        elif isinstance(label, int):
            label = [label,]
        for label_ in label:
            plt.figure()
            try:
                # if the geoplots package is installed
                self.cluster_centers_da.sel(cluster=label_).geo.plot(**kw)
            except:
                self.cluster_centers_da.sel(cluster=label_).plot(**kw)
            title = '{:4.1f}%'.format(self.get_cluster_fraction(label_)*100)
            plt.title(title)
