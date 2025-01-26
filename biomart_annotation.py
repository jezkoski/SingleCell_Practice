import pandas as pd
import scanpy as sc

def annotate_biomart_info(adata, biomart_file):
	"""Annotate adata object (in place) with gene position information from biomart.

	Args:
		adata: AnnData object
		biomart_file: path to gene position file from biomart. Includes columns ["ensembl_gene_id", "chromosome_name", "start_position","end_position"]
	"""

	biomart_locations = pd.read_csv(biomart_file, sep=" ")

	# when annotating adata object with data from dfs, the index has to match
	biomart_locations.index = biomart_locations["ensembl_gene_id"]

	adata.var["gene_name"] = adata.var_names
	adata.var_names = adata.var["gene_ids"]

	adata.var["chromosome"] = biomart_locations["chromosome_name"]
	adata.var["start"] = biomart_locations["start_position"]
	adata.var["end"] = biomart_locations["end_position"]
	# change gene names back as index
	adata.var_names = adata.var["gene_name"]



