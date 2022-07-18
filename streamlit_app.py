import streamlit as st
import pandas as pd
import numpy as np
import math
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
#from sklearn.metrics import silhouette_score
#from sklearn.decomposition import PCA
#import hdbscan

st.set_page_config(page_title='Clustering', 
                   #layout="wide", 
                   initial_sidebar_state='expanded')

# Data and variables
@st.cache
def get_data():
    vuln = gpd.read_file('data/tbl_vuln.gpkg').to_crs(3035)
    ind = pd.read_csv('data/ac_indicators_all.csv')
    char = pd.read_csv('data/pdo_characteristics.csv')
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    return(vuln, ind, char, world)

@st.cache
def calc_cluster(data, clmethod, clmethods):
    
    data_num = data.select_dtypes(np.number)
    
    #Calculate cluster predictions
    func = clmethods[clmethod][0]
    kwargs = clmethods[clmethod][1]
    
    data[clmethod] = func(**kwargs).fit_predict(data_num)
    data.sort_values(clmethod, inplace = True)
    data[clmethod] = data[clmethod].astype(str)

    #Create shapefile with cluster results
    cluster_shp = tbl_vuln.merge(data[['PDOid', clmethod]], on = 'PDOid')
    cluster_shp['geometry'] = cluster_shp['geometry'].centroid
    cluster_shp = cluster_shp.to_crs(4326)
    
    return(cluster_shp)

@st.cache(allow_output_mutation = True)
def cluster_map(cluster_shp):
    fig, ax = plt.subplots()
    world.to_crs(4326).plot(ax = ax, color = 'lightgrey', edgecolor = 'white', zorder = 0, lw = .5)
    for c, data in cluster_shp.groupby(m_select):
        label = f'{c}'
        p_color = colors_dict2[c]
        data.plot(color = p_color, marker = 'o', edgecolor = 'black', markersize = 15, 
                  linewidth = 0.4, ax = ax, label = label)
    ax.legend(loc = 'upper left')
    ax.set_facecolor('lightblue')
    ax.set_ylim(25, 55)
    ax.set_xlim(-20, 30)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return(fig)

@st.cache(allow_output_mutation = True)
def plot_bar(cluster_class, class_var):
    
    order = [
        'BWh Arid, desert, hot', 'BSh Arid, steppe, hot', 'BSk Arid, steppe, cold', 'Csa Temperate, dry summer, hot summer', 'Csb Temperate, dry summer, warm summer',
        'Cfa Temperate, no dry season, hot summer', 'Cfb Temperate, no dry season, warm summer', 'Dfa Cold, no dry season, hot summer', 'Dfb Cold, no dry season, warm summer',
        'ET Polar, tundra'
    ]
    
    cluster_perc = cluster_class.groupby([m_select, class_var], as_index = False)[class_var].size()
    cluster_perc['size_p'] = (cluster_perc['size'] / cluster_perc.groupby(m_select)['size'].transform('sum')) * 100
    cluster_perc = cluster_perc.pivot(index = m_select, values = 'size_p', columns = class_var).fillna(0)
    if class_var == 'koeppen':
        cluster_perc = cluster_perc[order]
    cluster_perc.sort_index(ascending = False, inplace = True)
    
    fig, ax = plt.subplots()
    p = cluster_perc.plot(kind = 'barh', stacked = True, legend = False, ax = ax, colormap='RdBu')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.1))
    ax.set_ylabel('Cluster')
    ax.set_xlim(0, 100)

    def format_label(x):
        if x > 5:
            return(str(int(round(x, 0))) + '%')
        else:
            return('')
    
    for container in p.containers:
        ax.bar_label(container, label_type = 'center', labels = [format_label(i) for i in container.datavalues])
        
    return(fig)

@st.cache(allow_output_mutation = True)
def plot_scatter(cluster_shp, clmethod, x_var, y_var):
    
    fig, ax = plt.subplots()
    sns.scatterplot(data = cluster_shp, x = x_var, y = y_var, hue = clmethod, ax = ax, palette = colors_dict2)
            
    ax.legend(loc = 'upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc = 'upper right', title = 'Clusters')
    
    return(fig)

@st.cache(allow_output_mutation = True)
def plot_boxes(tbl_all_info, box_vars, m_select):

    cols = 4
    rows = math.ceil(n_clusters/cols)
    params = {"sharey": True, "sharex": True}
    
    if rows == 1:
        fig, axs = plt.subplots(rows, cols, **params, figsize = (10, 3))
    else:
        fig, axs = plt.subplots(rows, cols, **params)

    for (g, data), ax in zip(vars_melt.groupby(m_select), axs.ravel()):

        sns.boxplot(ax = ax, data = data, color = colors_dict2[g], x = 'value', y = 'variable', showfliers = False)

        count = len(data.drop_duplicates(subset = 'PDOid'))
        ax.set_title(f'Cluster {g}\n{count} PDOs', weight = 'bold', size = 12)

        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.tight_layout()
    
    return(fig)

tbl_vuln, tbl_ind, tbl_class, world = get_data()
kmean_kwargs = {"init":"k-means++",
                "n_init":50,
                "max_iter":500,
                "random_state": 42}

#Start Layout
st.title('Clustering')

with st.sidebar:
    
    with st.form('Params'):
        mselect_container = st.container()
        k = st.slider('Number of Clusters', min_value = 2, max_value = 20, step = 1, value = 4)
        cl_select = st.multiselect('Cluster Variables', options = tbl_vuln.select_dtypes('number').columns.difference(['n_missing', 'lat', 'lon']),
                                   default = ['Exposure', 'Sensitivity', 'adaptive_capacity'])
        
        st.form_submit_button('Calculate!')
    
clmethods = {
    'kmeans': [cluster.KMeans, {'n_clusters': k, **kmean_kwargs}],
    'meanShift': [cluster.MeanShift, {'cluster_all': False, "bin_seeding": True}],
    'agg_ward': [cluster.AgglomerativeClustering, {"n_clusters": k, "linkage": 'ward', "affinity": 'euclidean'}],
    'agg_complete': [cluster.AgglomerativeClustering, {"n_clusters": k, "linkage": 'complete', "affinity": 'l1'}],
    'agg_average': [cluster.AgglomerativeClustering, {"n_clusters": k, "linkage": 'average', "affinity": 'l1'}],
    'agg_single': [cluster.AgglomerativeClustering, {"n_clusters": k, "linkage": 'single', "affinity": 'l1'}],
    'dbscan': [cluster.DBSCAN, {'eps':0.075, 'min_samples': 10}],
    #'hdbscan': [hdbscan.HDBSCAN, {'min_cluster_size':10}],
    'affinityPropagation': [cluster.AffinityPropagation, {'preference': -10, 'random_state': 5, 'damping': .7}],
    'spectral': [cluster.SpectralClustering, {"n_clusters": k, "eigen_solver": "arpack", "affinity": "nearest_neighbors"}]
}

with mselect_container:
    m_select = st.selectbox('Clustering Algorithm', options = clmethods.keys())
    
tbl_cluster = tbl_vuln[['PDOid', *cl_select]].dropna().reset_index(drop = True)

cluster_shp = calc_cluster(tbl_cluster, m_select, clmethods)
cluster_class = cluster_shp.merge(tbl_class, on = 'PDOid', how = 'left')

cmap = mpl.cm.get_cmap('Set1')
cluster_labels = cluster_shp[m_select].astype(int).unique()
n_clusters = len(cluster_labels)
colors = cmap(np.linspace(0, 1, n_clusters))
colors_dict = {cl: colors[i] for i, cl in enumerate(cluster_labels)}
colors_dict[-1] = [0, 0, 0, 0]
colors_dict2 = {str(cl): c for cl, c, in colors_dict.items()}

st.markdown("## Cluster map")
#Map with clusters
fig = cluster_map(cluster_shp)
st.pyplot(fig)

with st.expander('Cluster size'):
    cluster_size = cluster_shp.groupby(m_select, as_index = False).size()
    st.table(cluster_size.set_index(m_select))

st.markdown("---")
st.markdown("## Charakterisation of Clusters")

col1, col2, col3 = st.columns([1, .35, .35])
radio_options = cl_select + [i for i in ['Impacts', 'vulnerability'] if i not in cl_select]
class_var = col1.radio('Classification type', options = ['koeppen', 'landform'])
x_var = col2.radio('X-variable', options = radio_options, index = len(cl_select))
y_var = col3.radio('Y-variable', options = radio_options, index = cl_select.index('adaptive_capacity'))
    
c1, c2 = st.columns((1, 1))
with c1:

    #Barplot
    fig = plot_bar(cluster_class, class_var)
    st.pyplot(fig)

with c2:
    
    #Scatterplot
    fig = plot_scatter(cluster_shp, m_select, x_var, y_var)
    st.pyplot(fig)
    
st.markdown("---")
st.markdown("## Variable comparison amongst Clusters")

tbl_all_info = pd.merge(cluster_shp, tbl_ind, on = 'PDOid', how = 'left')

with st.form('Plot'):
    box_vars = st.multiselect('Select variables to compare', options = tbl_all_info.select_dtypes(np.number).columns.difference(['n_missing', 'lat', 'lon']),
                             default = ['Exposure', 'Sensitivity', 'adaptive_capacity'])
    st.form_submit_button('Plot variables!')
vars_melt = tbl_all_info.melt(id_vars = ['PDOid', m_select], value_vars = box_vars)

fig = plot_boxes(tbl_all_info, box_vars, m_select)
st.pyplot(fig)

st.markdown('### Average values of selected variables per Cluster')
centers = tbl_all_info.groupby(m_select)[box_vars].mean()
st.dataframe(centers.style.format('{:.2f}'))




