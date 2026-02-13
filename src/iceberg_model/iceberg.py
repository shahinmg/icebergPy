import numpy as np
import numpy.matlib
from scipy.interpolate import interp1d, interp2d
from scipy.spatial import cKDTree, KDTree
import xarray as xr
from math import ceil
from sklearn.linear_model import LinearRegression




class Iceberg:
    def __init__(self, length, ):
        
        length = self.length
        
        
        pass



    def keeldepth(self, L, method='barker'):
        """
        
    
        Parameters
        ----------
        L : Float
            Length of iceberg
        method : str
            Iceberg keel depth model. Default is barker
    
        Returns
        -------
        keel_depth : Float
            Deepest part of an iceberg
    
        """
        
        L_10 = np.round(L/10) * 10 # might have issues with this
        
        barker_La = L_10 <= 160
        hotzel_La = L_10 > 160
        
        # if nargin==1
        #     method = 'auto' # idk what this is from
        
        if method == 'auto':
            # not sure about this
            keel_depth_h = 3.78  * np.power(hotzel_La,0.63) # hotzel
            keel_depth_b = 2.91 * np.power(barker_La,0.71) # barker
            
            return keel_depth_b,keel_depth_h
            
        elif method == 'barker':
            keel_depth = 2.91 * np.power(L_10,0.71)
            
            return keel_depth
        
        elif method == 'hotzel':
            keel_depth = 3.78 * np.power(L_10,0.63)
        
            return keel_depth
        
        elif method == 'constant':
            keel_depth = 0.7 * L_10
            
            return keel_depth
        
        elif method == 'mean':
            
            keel_arr = np.ones(len(L_10),4)
    
            keel_arr[barker_La,0] = 2.91 * np.power(barker_La,0.71) # barker # feel like these should just be ind columns?
            keel_arr[hotzel_La,0] = 3.78 * np.power(hotzel_La,0.63) # hotzel
            keel_arr[:,1] = 2.91 * np.power(L_10,0.71)
            keel_arr[:,2] = 3.78 * np.power(L_10,0.63)
            keel_arr[:,3] = 0.7 * L_10
            
            mean = np.mean(keel_arr, axis=1)
            
            keel_depth = mean
            
            return keel_depth
    
    
    def barker_carea(self, L, keel_depth, dz, LWratio=1.62, tabular=200, method='barker'):
        # #
        # # calculates underwater cross sectional areas using Barker et al. 2004,
        # # and converts to length underwater (for 10 m thicknesses, this is just CrossArea/10) and sail area for K<200, 
        # # for icebergs K>200, assumes tabular shape
        # #
        # # [CArea, UWlength, SailA] = barker_carea(L)
        # #
        # # L is vector of iceberg lengths, 
        # # K is keel depth
        # # (if narg<2), then it calculates keel depths from this using keeldepth.m
        # # dz = layer thickness to use
        # # LWratio: optional argument, default is 1.62:1 L:W, or specify
        # #
        # # all variables in structure "icebergs"
        # #   CA is cross sectional area of each 10 m layer underwater
        # #   uwL is length of this 10 m underwater layer
        # #   Z is depth of layer
        # #   uwW calculated from length using length to width ratio of 1.62:1
        # # 
        # #   also get volumes and masses
        # #
        # # first get keel depth
        
        keel_depth = np.array([keel_depth])
        L = np.array([L])
        if keel_depth == None:
            keel_depth = self.keeldepth(L,'barker') # K = keeldepth(L,'mean');
            dz = 10
            LWratio = 1.62
        
        # table 5
        if dz == 10: # originally for dz=10 m layers
            a = [9.51,11.17,12.48,13.6,14.3,13.7,13.5,15.8,14.7,11.8,11.4,10.9,10.5,10.1,9.7,9.3,8.96,8.6,8.3,7.95]
            
            
            
            a = np.array(a).reshape((len(a),1))
            
            b = [25.9,107.5,232,344.6,457,433,520,1112,1125,853,931,1007,1080,1149,1216,1281,1343,1403,1460,1515]
            b = -1 * (np.array(b).reshape((len(b),1)))
            
        elif dz == 5:
    
            a = [9.51,11.17,12.48,13.6,14.3,13.7,13.5,15.8,14.7,11.8,11.4,10.9,10.5,10.1,9.7,9.3,8.96,8.6,8.3,7.95]
            a = np.array(a).reshape((len(a),1))
            
            b = [25.9,107.5,232,344.6,457,433,520,1112,1125,853,931,1007,1080,1149,1216,1281,1343,1403,1460,1515]
            b = -1 * (np.array(b).reshape((len(b),1)))
        
            # a_lin = a[9:,:]
            # b_lin = b[9:,:]
            # model = LinearRegression().fit(a_lin, b_lin)
            # r_sq = model.score(a_lin, b_lin)
            
            # mean = np.mean(np.diff(a_lin,axis=0))
            # a2 = np.arange(a_lin[-1][0],0,mean).reshape((-1,1))
            # b2 = model.predict(a2)
            
            # a_stack = np.vstack((a,a2[1:,:]))
            # b_stack = np.vstack((b,b2[1:,:]))
            
            
            aa = np.empty(a.T.shape)
            aa[0] = a[0]
            bb = np.empty(b.T.shape)
            bb[0] = b[0]
            
            for i in range(len(a)-1):
                aa[0,i+1] = np.nanmean(a[i:i+2,:])
                bb[0,i+1] = np.nanmean(b[i:i+2,:])
            
            # kz = keel_depth[0] # keel depth
            # kza = np.ceil(kz/dz) # layer index for keel depth
            # newa = np.empty((a.size*2,1)) #np.ceil(kz/dz) instead of 40?
            newa = np.empty((40,1)) #np.ceil(kz/dz) instead of 40?
            # if kza <= 40:    
            #     newa = np.empty((40,1)) #np.ceil(kz/dz) instead of 40?
            # elif kza > 40:
            #     newa = np.empty((int(kza),1)) #np.ceil(kz/dz) instead of 40?
    
            newa[:] = np.nan
            newb = newa.copy()
            
            newa[::2] = aa.T
            newa[1::2] = a
            
            newb[::2] = bb.T
            newb[1::2] = b
            
            a = newa/2
            b = newb/2
        
        a_s = 28.194; # for sail area table 4 barker et al 2004
        b_s = -1420.2;    
        
        
    
        
        
        
        # initialize arrays
        # icebergs.Z = dz:dz:500; icebergs.Z=icebergs.Z';
        # zlen = length(icebergs.Z);
        # temp = nan.*ones(zlen,length(L));  # 100 layers of 5-m each, so up to 500 m deep berg
        # temps = nan.*ones(1,length(L));  # sail area
        
        z_coord_flat = np.arange(dz,600+dz,dz) # deepest iceberg is defined here 
        z_coord = z_coord_flat.reshape(len(z_coord_flat),1)
        depth_layers = xr.DataArray(data=z_coord, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="Z")
        zlen = len(depth_layers.Z)
        # temp = nan.*ones(zlen,length(L))
        # need to make L an array
        
        temp = np.nan * np.ones((zlen, len(L)))
        temps = np.nan * np.ones((1, len(L)))
        
        
        # K_l200 = keel_depth[keel_depth<200] # might cause an issue?
        K_ltab = np.where(keel_depth<=tabular)[0] # get indices of keel_depth < tabular
        # if(~isempty(ind))
        if K_ltab.size != 0: # check if empty
            for i in range(len(K_ltab)):
                
                kz = keel_depth[i] # keel depth
                # dz_np = np.array([dz],dtype=np.float64)
                kza = np.ceil(kz/dz) # layer index for keel depth
                # kza = ceil(kz,dz) # layer index for keel depth
                
                for nl in range(int(kza)):
                    temp[nl,i] = a[nl] * L[K_ltab[i]] + b[nl]
                    
            temps[K_ltab] = a_s * L[K_ltab] + b_s
            
            if L < 65:
                temps[L<65] = 0.077 * np.power(L[L<65],2) # fix for L<65, barker 2004
        
        
        # then do icebergs D>200 for tabular
        K_gtab = np.where(keel_depth>tabular)[0]
        if K_gtab.size != 0:
            for i in range(len(K_gtab)):
                
                kz = keel_depth[i] # keel depth
                kza = np.ceil(kz/dz) # layer index for keel depth
                
                for nl in range(int(kza)):
                    # temp[nl,i] = a[nl] * L[K_g200[i]] + b[nl]
                    temp[nl,i] = L[K_gtab[i]] * dz
            
            temps[K_gtab] = 0.1211 * L[K_gtab] * keel_depth[K_gtab]
            
        
        cross_area = xr.DataArray(data=temp, coords = {"Z":z_coord_flat}, dims=["Z","X"], name="cross_area")
        # icebergs.uwL = temp./dz; 
        length_layers = xr.DataArray(data=temp/dz, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwL")
        
        # now use L/W ratio of 1.62:1 (from Dowdeswell et al.) to get widths I wonder if I can just get widths from Sid's data??
        widths = length_layers.values / LWratio 
        width_layers = xr.DataArray(data = widths, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwW")
        
        dznew = dz * np.ones(length_layers.values.shape);
        
        vol = dznew * length_layers.values * width_layers.values
        volume = xr.DataArray(data=vol, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwV")
        
        # I am ASSUMING everything is the same size. NEED TO CHECK when I get things running
        icebergs = xr.Dataset(data_vars={'depth':depth_layers,
                                         'cross_area':cross_area,
                                         'uwL':length_layers,
                                         'uwW':width_layers,
                                         'uwV':volume},
                              coords = {'Z': z_coord_flat}
                              )
    
        
        return icebergs
    
    def init_iceberg_size(self, L, dz=10, stability_method='equal',quiet=True):
        # # initialize iceberg size and shapes, based on length
        # # 
        # # given L, outputs all other iceberg parameters
        # # dz : specify layer thickness desired, default is 10m
        # #
        # # Updated to make stable using Wagner et al. threshold, Sept 2017
        # # either load in lengths L or specify here
        # #L = [100:50:1000]';
        # stablility method 'keel' or 'equal' 
        # keel changes keel depth, equal makes width and length equal
        
        keel_depth = self.keeldepth(L, 'barker')
        
        # now get underwater shape, based on Barker for K<200, tabular for K>200, and 
        ice = self.barker_carea(L, keel_depth, dz) # LWratio = 1.62 this gives you uwL, uwW, uwV, uwM, and vector Z down to keel depth
        
        # from underwater volume, calculate above water volume
        rho_i = 917 #kg/m3
        rat_i = rho_i/1024 # ratio of ice density to water density
        
        total_volume = (1/rat_i) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
        sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune
        
        waterline_width = L/1.62
        freeB = sail_volume / (L * waterline_width) # Freeboard height
        # length = L.copy()
        thickness = keel_depth + freeB # total thickness
        deepest_keel = np.ceil(keel_depth/dz) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
        # dz = dzS
        dzk = -1*((deepest_keel - 1) * dz - keel_depth) #
        
        # check if stable
        stability_thresh = 0.92 # from Wagner et al. 2017, if W/H < 0.92 then unstable
        stable_check = waterline_width / thickness[0]
        
        if stable_check > stability_thresh:
                ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
                ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
                ice['W'] = xr.DataArray(waterline_width, name='W')
                ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
                ice['L'] = xr.DataArray(np.float64(L),name='L')
                ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
                ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
                ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
                ice['dz'] = xr.DataArray(data=dz, name='dz')
                ice['dzk'] = xr.DataArray(data=dzk, name='dzk')
                
                return ice
        
        
        if stable_check < stability_thresh:
            # Not sure when to use either? MATLAB code has if(0) and if(1) for 'keel' and 'equal'
            if stability_method == 'keel':
                # change keeldepth to be shallower
                # if quiet == False:
                #     print(f'Fixing keel depth for L = {L} m size class')
                    
                
                diff_thick_width = thickness - waterline_width # Get stable thickness
                keel_new = keel_depth - rat_i * diff_thick_width # change by percent of difference
                
                ice = self.barker_carea(L,keel_new,dz)
                total_volume = (1/rat_i) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
                sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune
                waterline_width = L/1.62 
                freeB = sail_volume / (L * waterline_width) # Freeboard height
                # length = L.copy()
                thickness = keel_depth + freeB # total thickness
                deepest_keel = np.ceil(keel_depth/dz) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
                # dz = dzS
                dzk = -1*((deepest_keel - 1) * dz - keel_depth) #
                stability = waterline_width/thickness
                
                ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
                ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
                ice['W'] = xr.DataArray(waterline_width, name='W')
                ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
                ice['L'] = xr.DataArray(np.float64(L),name='L')
                ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
                ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
                ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
                ice['dz'] = xr.DataArray(data=dz, name='dz')
                ice['dzk'] = xr.DataArray(data=dzk, name='dzk')
                
                return ice
            
            elif stability_method == 'equal':
                # change W to equal L, recalculate volumes
                if quiet == False:
                    print(f'Fixing width to equal L, for L = {L} m size class')
                # use L:W ratio of to make stable, set so L:W makes EC=EC_thresh
                
                width_temporary = stability_thresh * thickness[0]
                lw_ratio = np.floor((100*L)/width_temporary)/100 # round down to hundredth place
                
                ice = self.barker_carea(L, keel_depth, dz, LWratio=lw_ratio)
                
                total_volume = (1/rat_i) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
                sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune
                waterline_width = L / lw_ratio 
                freeB = sail_volume / (L * waterline_width) # Freeboard height
                # length = L.copy()
                thickness = keel_depth + freeB # total thickness
                deepest_keel = np.ceil(keel_depth/dz) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
                # dz = dzS
                dzk = -1*((deepest_keel - 1) * dz - keel_depth) #
    
                
                ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
                ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
                ice['W'] = xr.DataArray(waterline_width, name='W')
                ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
                ice['L'] = xr.DataArray(np.float64(L),name='L')
                ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
                ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
                ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
                ice['dz'] = xr.DataArray(data=dz, name='dz')
                ice['dzk'] = xr.DataArray(data=dzk, name='dzk')
                EC = ice.W/ice.TH
                
                if EC < stability_thresh:
                    raise Exception("Still unstable, check W/H ratios")
                
                return ice