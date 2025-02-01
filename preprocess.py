import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation


def basic_filters(adata):
	"""Make basic sc-analysis filtering steps. Changes var_names to adata.var["SYMBOL"] and saves the index to adata.var["gene_id"].

	min_cells=3, min_genes=200, mitochondrial, ribosomal and HB genes marked, qc-metrics calculated
	"""

	adata.var["gene_id"] = adata.var_names
	adata.var_names = list(adata.var["symbol"])
	adata.var_names_make_unique()
	#change numerical batch values to categories
	adata.obst["Batch"] = adata.obs["Batch"].astype("category")

	sc.pp.filter_genes(adata, min_cells=3)
	sc.pp.filter_cells(adata, min_genes=200)
	print(f"Number of cells after filtering low quality cells: {adata.n_obs}")

	adata.var["mt"] = adata.var_names.str.startswith("MT-")
	adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
	adata.var["hb"] = adata.var_names.str.startswith(("^HB[^(P)]"))

	sc.pp.calculate_qc_metrics(
		adata, qc_vars=["mt"], percent_top=[20], log1p=True, inplace=True
	)


def is_outlier(adata, metric: str, nmads: int):
	"""Outlier function from scanpy tutorial
	"""

	M = adata.obs[metric]
	outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
		np.median(M) + nmads * median_abs_deciation(M) < M
	)

	return outlier

def calculate_outliers(adata):
	"""Calculate outliers function from scanpy tutorial
	"""

	adata.obs["outlier"] = (
		is_outlier(adata, "log1p_total_counts", 5)
		| is_outlier(adata, "log1p_n_genes_by_counts", 5)
		| is_outlier(adata, "pct_counts_in_top_20_genes", 5)
	)

	adata.obs.outlier.value_counts()

	adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3)
	adata.obs.mt_outlier.value_counts()

	print(f"Total number of cells: {adata.n_obs}")



