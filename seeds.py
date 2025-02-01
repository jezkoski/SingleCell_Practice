import scanpy as sc
import pandas as pd



def preprocess_for_seeds(adata, gene_subset):
	"""Preprocess: normalize+log1p and scale the data fro seed scoring. Only for a subset of the
	 genes (marker genes) from a pre-defined list.

	Args:
		adata: AnnData object
		gene_subset: list of marker genes

	Return:
		normalized: normalized, log-transformed and scaled subset of the original adata containin only the marker genes
	"""

	normalized = adata.copy()
	sc.pp.normalize_total(normalized, target_sum=1e4)
	sc.pp.log1p(normalized)

	normalized = normalized[:, gene_subset].copy()
	sc.pp.scale(normalized)

	return normalized


def get_score(normalized_adata, gene_set):
	"""Returns the score per cell given a dictionary of + and - genes. From scvi-tools.

	Args:
		normalized_data: anndata dataset that has been log normalized and scaled to mean 0, std1
		gene_set: a dictionary with two keys: 'positive' and 'negative'
			each key should contain a list of genes
			for each gene in gene_set['positive'], its expression will be added to the score
			for each gene in gene_set['negative'], its expression will be substracted from its score

	Returns:
		score: array of length n_cells containing the score per cell
	"""

	score = np.zeros(normalized_adata.n_obs)
	for gene in gene_set["positive"]:
		expression = np.array(normalized_adata[:, gene].X)
		score += expression.flatten()
	for gene in gene_set["negative"]:
		expression = np.array(normalized_adata[:, gene].X)
		score -= expression.flatten()

	return score

def get_cell_mask(normalized_adata, gene_set, X):
	"""Calculates the score per cell for a list of genes, then returns a mask for the cells with the highest X scores.

	Args:
		normalized_adata: anndata dataset that has been log normalized and scaled to mean 0 std1
		gene_set: a dictionary with two keys: 'positive' and 'negative'
                        each key should contain a list of genes
                        for each gene in gene_set['positive'], its expression will be added to the score
                        for each gene in gene_set['negative'], its expression will be substracted from its score
		X: the number of scores (seeds) included
	Returns:
		mask: Mask for the cells with the top X scores over the entire dataset
	"""

	score = get_score(normalized_adata, gene_set)

	cell_idx = score.argsort()[-X:]
	mask = np.zeros(normalizeD_adata.n_obs)
	mask[cell_idx] = 1

	return mask.astype(bool)


def read_aximuth(azimuth):
	"""Reads and preprocesses the azimuth data for the cell type markers.
	"""

	azimuth_markers = pd.read_csv(azimuth, sep="\t")
	azimuth_markers["Markers"] = azimuth_markers.Markers.apply(lambda x: x.strip("").split(", "))

	return azimuth_markers


def remove_problematic_genes(genelist):
	"""Removes genes that were found to be problematic in the analysis.
	"""

	if "HRASLS2" in genelist:
		genelist.remove("HRASLS2")

	return genelist

def get_flat_genelist(data):
	"""Get list of all the marker genes"""

	gene_flatlist = [item for sublist in data["Markers"] for item in sublist]
	gene_flatlist = list(set(gene_flatlist))
	gene_flatlist = remove_problematic_genes(gene_flatlist)

	return gene_flatlist

def get_seeds(normalized, markers):
	"""Gets the cell mask and score for each cell type.

	Args:
		normalized: anndata dataset that has been log normalized and scaled to mean 0 std1
		markers: cell type markers

	Returns:
		seed_labels: labels for all the cells in the object. Unknown label for those that are not 'seeds'
	"""


	set_to_unknown = False

	for i, row in markers.iterrows():
		label = row["Label"]
		genelist = row["Markers"]

		genelist = remove_problematic_genes(genelist)

		geneset = {"positive" : genelist, "negative": []}
		print(f"{label} {geneset}")

		mask = get_cell_mask(normalized, geneset)

		if set_to_unknown==False:
			seed_labels = np.array(mask.shape[0] * ["Unknown"]
			set_to_unknown = True

		seed_labels[mask] = label

	return seed_labels


