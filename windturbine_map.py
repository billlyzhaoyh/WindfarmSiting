#visulaise windturbine in a world map or respective country map

#Default plot all wind turbines on a world map

def plot_windturbine(csv_path,country='world'):
	import pandas as pd
	import matplotlib.pyplot as plt
	import descartes
	import geopandas as gpd
	from shapely.geometry import Point, Polygon
	import os
	base_dir = '/Users/billyzhaoyh/Desktop/fourth_year_project/Cleanup/Maps'
	df = pd.read_csv(csv_path, sep=",")
	viz_map=gpd.read_file(os.path.join(base_dir, country+'.shp'))
	if country=='world':
		df_country=df
	else:
		df_country=df.loc[df['Country'] == country]
	geometry_country=[Point(xy) for xy in zip(df_country['Longitude(x)'],df_country['Latitude(y)'])]
    geometry_country[:3]
    crs = {'init':'epsg:4326'}
    geo_df=gpd.GeoDataFrame(df_country,crs=crs,geometry=geometry_country)
    fig,ax=plt.subplots(figsize=(60,60))
    country_map.plot(ax=ax,alpha=0.4,color='grey')
    geo_df.plot(ax=ax,markersize=10,color='red',marker="o",label="wind turbines")
    plt.legend(prop={'size':60})

