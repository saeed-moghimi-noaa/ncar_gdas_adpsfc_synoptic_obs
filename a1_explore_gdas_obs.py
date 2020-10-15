import glob
import os
import pandas as pd
import matplotlib.pyplot as plt 

import numpy as np
import matplotlib.cm as cm


#windrose related
from windrose import WindroseAxes
from windrose import plot_windrose
import windrose


#map related
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER,
                                   LATITUDE_FORMATTER)
import cartopy.feature as cfeature 
#
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText


from cartopy.io.img_tiles import Stamen



def make_map_cartopy(projection=ccrs.PlateCarree()):                                                                                                                                        
                                                                                           
    """                                                                          
    Generate fig and ax using cartopy                                                                    
    input: projection                                                                                    
    output: fig and ax                             
    """                                  
    alpha = 0.5                                        
    subplot_kw = dict(projection=projection)                        
    fig, ax = plt.subplots(figsize=(12, 8),                           
                           subplot_kw=subplot_kw)   
    gl = ax.gridlines(draw_labels=True)                                 
    gl.xlabels_top = gl.ylabels_right = False 
    gl.xformatter = LONGITUDE_FORMATTER                        
    gl.yformatter = LATITUDE_FORMATTER                                
                                    
    if False:
        # Put a background image on for nice sea rendering.             
        ax.stock_img()                                   
                                                              
        # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
        states_provinces = cfeature.NaturalEarthFeature(                      
            category='cultural',                  
            name='admin_1_states_provinces_lines',
            scale='100m',           
            facecolor='none')        

        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
                                                                                                                                                                                        
        ax.add_feature(cfeature.LAND,zorder=0,alpha=alpha)          
        ax.add_feature(cfeature.COASTLINE,zorder=1,alpha=alpha)
        ax.add_feature(cfeature.BORDERS,zorder=1,alpha=2*alpha)
                           
        ax.add_feature(states_provinces, edgecolor='gray',zorder=1)
                                                              
        # Add a text annotation for the license information to the
        # the bottom right corner.                                            
        text = AnchoredText(r'$\mathcircled{{c}}$ {}; license: {}'
                            ''.format(SOURCE, LICENSE),
                            loc=4, prop={'size': 9}, frameon=True)                                    
        ax.add_artist(text)                                                                           
    else:
        tiler = Stamen('terrain-background')
        ax.add_image(tiler, 6)
        ax.coastlines('10m')                                        
    
    
    ax.set_ylim(18.752631 , 32)  #lat limits           
    ax.set_xlim(46.879415 , 70.123184)  #lon limits   
    
    return fig, ax


def untar_gz():
    dirs = glob.glob('gdas*.txt')

    for dir1 in dirs:
        os.system('cd  ' + dir1+ '; gunzip *.gz; cd ..')


def obs_data2csv():
    f_synop_all = '../gdas.ADPSFC.all.csv'
    if not os.path.exists(base_url + fname):
        del synop
        # untar gunzip files
        dirs  = glob.glob('gdas*.txt')
        for dir1 in dirs:
            files = glob.glob(dir1 + '/gdas*.txt')
            for file1 in files:
                tmp = pd.read_csv(file1,  delimiter=r"\s+",skiprows=2)   #header=[0,1],
                if 'synop' not in locals():
                    synop = tmp
                else:
                    synop  = pd.concat([synop,tmp], axis=0)          

        synop.to_csv(f_synop_all)
    else:
        synop = pd.read_csv(f_synop_all)

    return synop


f_synop_all = 'gdas.ADPSFC.all.csv'
syn = pd.read_csv(f_synop_all,parse_dates=['YYYYMMDDHHMM'])

syn['wind_math_direction'] = 270 - syn.WDIR
syn.wind_math_direction[syn.wind_math_direction<0] = syn.wind_math_direction[syn.wind_math_direction<0] + 360

syn['wind_x'] = syn.WSPD * np.cos(syn.wind_math_direction)
syn['wind_y'] = syn.WSPD * np.sin(syn.wind_math_direction)

grps = syn.groupby('BBSSS')

#for windrose
bins = np.arange(0.0, 15, 2.5)



#1) OBBI Bahrain International Airport   WMO id: 41150
OBBI_lat, OBBI_lon =  26.2708333333, 50.6336111111

st41150 = grps.get_group(41150).set_index('YYYYMMDDHHMM').sort_index()
st41150.lat = OBBI_lat
st41150.lon = OBBI_lon
st41150.label =  'Bahrain International Airport (OBBI; WMO id:41150)'

st41150 [st41150['SPD(M/S)'] < -100] = np.nan
st41150 [st41150['SPD(M/S)'] >  100] = np.nan
st41150.dropna(inplace = True)
st41150['SPD(M/S)'].plot(title=st41150.label)

ax = WindroseAxes.from_ax()
plt.gcf().suptitle(st41150.label)
ax.bar(direction = st41150.WDIR, var = st41150.WSPD, blowto=True, bins = bins, opening=0.8, edgecolor='white')
ax.set_legend()




#2) Muscat International Airport (OOMS; WMO id: 41256)
OOMS_lat, OOMS_lon = 23.5927777778, 58.2841666667

plt.figure()
st41256 = grps.get_group(41256).set_index('YYYYMMDDHHMM').sort_index()
st41256.lat = OOMS_lat
st41256.lon = OOMS_lon
st41256.label = 'Muscat International Airport (OOMS; WMO id: 41256)'

st41256 [st41256['SPD(M/S)'] < -100] = np.nan
st41256 [st41256['SPD(M/S)'] >  100] = np.nan
st41256.dropna(inplace = True)
st41256['SPD(M/S)'].plot(title=st41256.label )


ax = WindroseAxes.from_ax()
plt.gcf().suptitle(st41256.label)
ax.bar(direction = st41256.WDIR, var = st41256.WSPD, blowto=True, bins = bins, opening=0.8, edgecolor='white')
ax.set_legend()


#save stations csv files
st41256.to_csv(st41256.label.replace(' ','_').replace(';','_').replace(':','_').replace('(','').replace(')','')+'.csv')
st41150.to_csv(st41150.label.replace(' ','_').replace(';','_').replace(':','_').replace('(','').replace(')','')+'.csv')


#bins = np.arange(0.01, 10, 1)
#plot_windrose(direction_or_df= st41150.WDIR,  var = st41150.WSPD, kind="contour",blowto=True, bins=bins, cmap=cm.hot, lw=3, rmax=2000)
#ax, param = plot_windrose(direction_or_df= st41150.WDIR,  var = st41150.WSPD, kind="pdf"    ,blowto=True, bins=bins, cmap=cm.hot, lw=3, rmax=2000)

############### plot map
map_fig, map_ax = make_map_cartopy()
# Inset axe it with a fixed size
wrax_st41256 = inset_axes(map_ax,
        width=1.2,                             # size in inches
        height=1.2,                            # size in inches
        loc='center',                        # center bbox at given position
        bbox_to_anchor=(st41256.lon, st41256.lat), # position of the axe
        bbox_transform=map_ax.transData,    # use data coordinate (not axe coordinate)
        axes_class=windrose.WindroseAxes,    # specify the class of the axe
        )

wrax_st41256.bar(direction = st41256.WDIR, var = st41256.WSPD, blowto=True, bins = bins,  opening=0.8, edgecolor='white')
#wrax_st41256.text(1,90,st41256.label)

wrax_st41150 = inset_axes(map_ax,
        width=1.2,                             # size in inches
        height=1.2,                            # size in inches
        loc='center',                        # center bbox at given position
        bbox_to_anchor=(st41150.lon, st41150.lat), # position of the axe
        bbox_transform=map_ax.transData,    # use data coordinate (not axe coordinate)
        axes_class=windrose.WindroseAxes,    # specify the class of the axe
        )

wrax_st41150.bar(direction = st41150.WDIR, var = st41150.WSPD, blowto=True, bins = bins,  opening=0.8, edgecolor='white')
#wrax_st41150.text(180,-3000,st41150.label)

for ax in [wrax_st41150, wrax_st41256]:
  ax.tick_params(labelleft=False, labelbottom=False)

leg = wrax_st41256.set_legend(bbox_to_anchor=(-3, -0.8),title='Speed [m/s]')
map_fig.suptitle('Wind direction: Blow to')
map_fig.savefig('../wind_roses.png', dpi = 300)






plt.show()






