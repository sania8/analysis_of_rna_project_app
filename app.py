import streamlit as st
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu
import matplotlib.pyplot as plt
import matplotlib

# Disable the warning about pyplot's global use
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set the page layout
st.set_page_config(layout="wide")

# Title
st.title("Data Analysis of Single-cell RNA-seq Experiment")
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Create columns for layout
col1, col2, col3 = st.columns([2,2,2])

# Column 1: Project explanation
with col1:
    st.markdown("""
            <h4>Project Overview</h4>
            <ol style="font-size:10px;">
                <li>Read gene expression matrix to determine cell and gene count.</li>
                <li>Plot 1 to visualize gene distribution across cells.</li>
                <li>Plot 2 to assess gene spread, filtering out mitochondrial RNA for quality control.</li>
                <li>Filter cells based on threshold values.</li>
                <li>Plot 3 to analyze variation in gene expression across cells.</li>
                <li>Select features (e.g., gene expressions between 0.02 and 4) for further analysis.</li>
                <li>Plot 4 to perform PCA for dimensionality reduction.</li>
                <li>Use Leiden Algorithm to cluster cells based on gene expression patterns presented in Plot 5</li>
            </ol>
    """, unsafe_allow_html=True)

# Column 2: File upload interface
with col2:
    st.markdown("""
             <h4>Upload file</h4>

                """,unsafe_allow_html=True)
   
    h5_file = st.file_uploader("Upload Filtered Feature Barcode Matrix (HDF5)", type="h5")
    # Display information points within a light blue box
    st.markdown("""
    <div style="background-color:#ECEEFF; padding: 5px; border-radius: 5px; display: inline-block;">
        <p style="font-size: 2px; text-align:center; color:black;"> 
            <h6>The dataset used was <a href="https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k">PBMC from a healthy donor - granulocytes removed through cell sorting (10k)<h6><br>
            <ul>
           <li> Click <a href="https://cf.10xgenomics.com/samples/cell-arc/1.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5">here</a> to download the (.h5) file and use to analyze results.<br></li>
            <li>Click on my <a href="https://github.com/sania8/Data-Analysis-of-Single-cell-RNA-seq-Experiment.git">GitHub profile</a> github repository to understand better.</li></ul>
        </p>
    </div>
""", unsafe_allow_html=True)

# Column 3: Analysis and results display
with col3:
    st.markdown("""
             <h4>Results</h4>

                """,unsafe_allow_html=True)
    if h5_file:
        # Save uploaded file to a temporary path
        with open("temp_filtered_feature_bc_matrix.h5", 'wb') as f:
            f.write(h5_file.getvalue())

        # Load the data
        mdata = mu.read_10x_h5("temp_filtered_feature_bc_matrix.h5")
        mdata.var_names_make_unique()

        # Process RNA data
        rna = mdata.mod['rna']

        # Quality control
        rna.var['mt'] = rna.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        # Plot before filtering
        st.markdown("""<b>Plot before filtering out cells(Plot 1)</b>
                    """,unsafe_allow_html=True)
        fig_before_filtering, axs_before_filtering = plt.subplots(1, 3, figsize=(15, 5))
        sc.pl.violin(rna, 'n_genes_by_counts', jitter=0.4, multi_panel=False, ax=axs_before_filtering[0])
        sc.pl.violin(rna, 'total_counts', jitter=0.4, multi_panel=False, ax=axs_before_filtering[1])
        sc.pl.violin(rna, 'pct_counts_mt', jitter=0.4, multi_panel=False, ax=axs_before_filtering[2])
        st.pyplot(fig_before_filtering)
        plt.clf()  # Clear the figure

        # Filtering
        mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
        mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))
        mu.pp.filter_obs(rna, 'total_counts', lambda x: x < 15000)
        mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 20)

        # Normalization
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)

        # Plot after filtering
        st.markdown("""<b>Plot after filtering out cells having high mitochondrial genes(Plot 2)</b>
                    """,unsafe_allow_html=True)
        fig_after_filtering, axs_after_filtering = plt.subplots(1, 3, figsize=(15, 5))
        sc.pl.violin(rna, 'n_genes_by_counts', jitter=0.4, multi_panel=False, ax=axs_after_filtering[0])
        sc.pl.violin(rna, 'total_counts', jitter=0.4, multi_panel=False, ax=axs_after_filtering[1])
        sc.pl.violin(rna, 'pct_counts_mt', jitter=0.4, multi_panel=False, ax=axs_after_filtering[2])
        st.pyplot(fig_after_filtering)
        plt.clf()  # Clear the figure
        # Feature selection
        sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
        
        #Plot highly variable genes
        st.markdown("""<b>High Variable genes across cells (Plot 3)</b>
                    """,unsafe_allow_html=True)
        sc.pl.highly_variable_genes(rna)
        st.pyplot()  # Display the plot
        
        # Additional Analysis: PCA
        st.markdown("""<b>Principal Component Analysis.(Plot 4)</b>
                    """,unsafe_allow_html=True)
        sc.tl.pca(rna, svd_solver='arpack')
        sc.pl.pca(rna, color=['CD2', 'CD79A', 'KLF4', 'IRF8'])
        st.pyplot()  # Display the plot
        #UMAP plot
        st.markdown("""<b>Identifying distinct cell clusters characterized by elevated gene expression levels for targeted analysis.(Plot 5)</b>
                    """,unsafe_allow_html=True)
        sc.pl.pca_variance_ratio(rna, log=True)
        sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)
        sc.tl.leiden(rna, resolution=.5)
        sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)
        sc.pl.umap(rna, color="leiden", legend_loc="on data")
        st.pyplot()
        
        # Additional Analysis: Other steps like scaling, clustering, etc.
        # (You can add more analysis steps here if needed)
        
        st.success("Analysis completed and results displayed.")
    else:
        st.warning("Please upload the required HDF5 file to proceed.")
