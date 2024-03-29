import streamlit as st
import pandas as pd
import numpy as np
import math
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
from copy import deepcopy
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
    
    preds = func(**kwargs).fit_predict(data_num)
    
    data[clmethod] = preds
    data.sort_values(clmethod, inplace = True)
    data[clmethod] = data[clmethod].astype(str)

    #Create shapefile with cluster results
    cluster_shp = tbl_vuln.merge(data[['PDOid', clmethod]], on = 'PDOid')
    cluster_shp['geometry'] = cluster_shp['geometry'].centroid
    cluster_shp = cluster_shp.to_crs(4326)
    
    if clmethod in ['kmeans', 'meanShift', 'affinityPropagation']:
        fit = func(**kwargs).fit(data_num)
        centers = pd.DataFrame(fit.cluster_centers_, columns = data_num.columns)
    
    elif clmethod in ['agg_ward', 'agg_complete', 'agg_average', 'agg_single', 'spectral', 'dbscan', 'hdbscan']:
        clf = sklearn.neighbors.NearestCentroid()
        clf.fit(data_num, preds)
        centers = pd.DataFrame(clf.centroids_, columns = data_num.columns)
        
    else:
        centers = pd.DataFrame(columns = data_num.columns)
        
    centers['sum'] = centers['Exposure'] + centers['Sensitivity'] - centers['adaptive_capacity']
    centers['Vulnerability'] = pd.cut(centers['sum'], [-1, .4, .8, 1.2, 1.6, 2], labels = ['Very low', 'Low', 'Medium', 'High', 'Very high'])
    centers[clmethod] = np.arange(len(centers)).astype(str)
    
    cluster_shp = cluster_shp.merge(centers[[clmethod, 'Vulnerability']], on = clmethod, how = 'left')
    cluster_shp[clmethod] = pd.Categorical(cluster_shp[clmethod], categories = np.sort(cluster_shp[clmethod].unique()))
    
    return(cluster_shp, centers)

@st.cache
def validate_kmeans(data, kmean_kwargs):
    
    data.dropna(inplace = True)
    
    #Determine nr of clusters
    sse = []
    s_coef = []
    for k in range(1, 15):
        kmeans = cluster.KMeans(n_clusters=k, **kmean_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        
        if k > 1:
            score = silhouette_score(data, kmeans.labels_)
            s_coef.append(score)
        else:
            s_coef.append(np.nan)
            
    return(sse, s_coef)

@st.cache(allow_output_mutation = True)
def cluster_map(cluster_shp, col):
    fig, ax = plt.subplots()
    world.to_crs(4326).plot(ax = ax, color = 'lightgrey', edgecolor = 'white', zorder = 0, lw = .5)
    
    #Without copying, this part causes an output-modification warning
    shp_copy = deepcopy(cluster_shp)
    for c, data in shp_copy.groupby(col):
        if len(data) > 0:
            label = f'{c}'
            p_color = colors_dict[c]
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
def plot_scatter(cluster_shp, clmethod, col):
    
    fig, ax = plt.subplots()
    sns.scatterplot(data = cluster_shp, x = 'Impacts', y = 'adaptive_capacity', hue = col, ax = ax, palette = colors_dict)
            
    ax.legend(loc = 'upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # handles, labels = ax.get_legend_handles_labels()
    # # sort both labels and handles by labels
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # ax.legend(handles, labels, loc = 'upper right', title = 'Clusters')
    
    return(fig)

@st.cache(allow_output_mutation = True)
def plot_boxes(tbl_all_info, box_vars, m_select):
    
    n_clusters = len(tbl_all_info[m_select].unique())
    cols = 4
    rows = math.ceil(n_clusters/cols)
    params = {"sharey": True, "sharex": True}
    
    if rows == 1:
        fig, axs = plt.subplots(rows, cols, **params, figsize = (10, 3))
    else:
        fig, axs = plt.subplots(rows, cols, **params)

    for (g, data), ax in zip(vars_melt.groupby(m_select), axs.ravel()):

        sns.boxplot(ax = ax, data = data, x = 'value', y = 'variable', showfliers = False)

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
        k = st.slider('Number of Clusters', min_value = 2, max_value = 15, step = 1, value = 4)
        cl_select = st.multiselect('Cluster Variables', options = tbl_vuln.select_dtypes('number').columns.difference(['n_missing', 'lat', 'lon']),
                                   default = ['Exposure', 'Sensitivity', 'adaptive_capacity'])
        
        st.form_submit_button('Calculate!')
        
    disp_option = st.radio('Display option', options = ['Clusters', 'Vulnerability'], index = 1)
    class_var = st.radio('Classification type', options = ['koeppen', 'landform'])
    
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
    
plot_col = m_select if disp_option == 'Clusters' else 'Vulnerability'
    
st.markdown(f'Algorithm: **{m_select}**')
if not 'n_clusters' in clmethods[m_select][1].keys():
    st.write('Cluster number is chosen automatically by this algorithm, therefore ignore parameter k')
    
tbl_cluster = tbl_vuln[['PDOid', *cl_select]].dropna().reset_index(drop = True)

cluster_shp, centers = calc_cluster(tbl_cluster, m_select, clmethods)
cluster_class = cluster_shp.merge(tbl_class, on = 'PDOid', how = 'left')

cmap = mpl.cm.get_cmap('Set1')
labels = cluster_shp[plot_col].cat.categories
n_labs = len(labels)

colors = cmap(np.linspace(0, 1, n_labs))
colors_dict = {str(cl): colors[i] for i, cl in enumerate(labels)}

st.markdown("## Cluster map")
st.write('This map shows the spatial distribution of the resulting clusters across Europe. The color indicates the cluster or the vulnerability level, depending on the selected display option in the sidebar.')
#Map with clusters
fig = cluster_map(cluster_shp, plot_col)
st.pyplot(fig)
      
st.markdown("---")
st.markdown("## Variable comparison amongst Clusters")
st.write('The boxplot below shows the distribution of different variables across the various clusters. Use the selectbox to choose the variables that should be compared. By clickingn on *Plot variables!* the plot will be updated.')

tbl_all_info = pd.merge(cluster_shp, tbl_ind, on = 'PDOid', how = 'left')

with st.form('Plot'):
    box_vars = st.multiselect('Select variables to compare', options = tbl_all_info.select_dtypes(np.number).columns.difference(['n_missing', 'lat', 'lon']),
                              default = ['Exposure', 'Sensitivity', 'adaptive_capacity'])
    st.form_submit_button('Plot variables!')
    
vars_melt = tbl_all_info.melt(id_vars = ['PDOid', m_select], value_vars = box_vars)

fig = plot_boxes(tbl_all_info, box_vars, m_select)
st.pyplot(fig)

st.markdown('### Cluster centroids')
st.write('The table below shows the centroid of the selected variables for each cluster as well as the resulting vulnerability for each cluster. The sum is given by *Exposure + Sensitivity - adaptive_capacity*. The highest possible value is 2 which would also correspond to the highest vulnerability. To calculate the vulnerability, the sum is classified into 5 equally sized groups (0 - 0.4, 0.4 - 0.8, 0.8 - 1.2, 1.2 - 1.6, 1.6 - 2.0) which correspond to the five vulnerability levels (very low, low, medium, high, very high).')
st.table(centers.set_index(m_select))

st.markdown('### PDOs per vulnerability level')
st.write('The plot below shows the number of PDOs for each vulnerability level')
st.table(cluster_shp.groupby('Vulnerability', as_index = False).size().rename(columns = {0: 'n'}))

st.markdown("---")
st.markdown("## Scatterplot")
st.write('In the scatterplot below, each PDO is represented by a point and the color indicates the cluster or vulnerability level. Use the buttons in the sidebar to change between the two display options.')

fig = plot_scatter(cluster_shp, m_select, plot_col)
st.pyplot(fig)

st.markdown("---")
st.markdown("## Barchart")
st.write('The barchart below shows the distribution of Koeppen-Geiger climate classes or different landform types within the different clusters, e.g. how many PDOs within a cluster fall into a given climate class. Use the classification type option in the sidebar to change between the two variables.')

fig = plot_bar(cluster_class, class_var)
st.pyplot(fig)

if m_select == 'kmeans':
    st.markdown("---")
    st.markdown("## Kmeans Validation")
    st.write('In the case of Kmean-Clustering, it is possible to get an approximation of the optimum number of clusters by looking at the Inertia (left plot, often called elbow method) or the Silhouette coefficient (right plot). The optimum number of clusters is usually around the point after which the SSE or inertia starts decreasing in a linear fashion or that maximizes the Silhouette Coefficient.')
    sse, s_coef = validate_kmeans(tbl_vuln[cl_select], kmean_kwargs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
    
    ax1.plot(range(len(sse)), sse, marker = 'o', linestyle = '--')
    ax1.axvline(x = k, color = 'red')
    ax1.set_title('SSE')
    
    ax2.plot(range(len(s_coef)), s_coef, marker = 'o', linestyle = '--')
    ax2.axvline(x = k, color = 'red')
    ax2.set_title('Silhouette coefficient')
    
    st.pyplot(fig)
