import anndata as ad
from anndata import AnnData

def concat_adata(adatas: list, merge=None, uns_merge=None) -> AnnData:
	"""Concatenate multiple AnnData objects.

	Args:
		adatas: list of adata objects to merge
		merge: how elements not aligned to the axis being concatenated along are selected.
		uns_merge: how elements of .uns are selected

	Returns:
		merge: merged adata object
	"""

	merge = ad.concat(adatas, merge=merge, uns_merge=uns_merge)

	return merge
