import anndata as ad
from anndata import AnnData

def create_adata_object(X_arr, obs_names, var_names) -> AnnData:
	"""Create an adata object from count matrix X_arr and cell and gene names.

	Args:
		X_arr: count matrix
		obs_names: array of cell names
		var_names: array of gene names

	Returns:
		adata: created AnnData object
	"""

	adata = ad.AnnData(X_arr)
	adata.obs_names = obs_names
	adata.var_names = var_names

	return adata
