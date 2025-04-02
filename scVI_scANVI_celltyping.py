import scanpy as sc
import scvi
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import seaborn as sns

print(f'scVI version: {scvi.__version__}')
print(f'torch version: {torch.__version__}')
use_gpu = True
device = 'cpu'

model_params = {
	'n_hidden': 128,
	'n_latent': 30,
	'n_layers': 2,
	'max_epochs_scVI': 500,
	'max_epochs_scANVI'; 200
}

plan_kwargs = {
	'reduce_lr_on_plateau': True,
	'lr_patience': 8,
	'lr_factor': 0.1
}

def check_cuda():
	if torch.cuda.is_available():
		print(f'CUDA version: {torch.version.cuda}')
		print(torch.zeros(1).cuda())
		cuda_id = torch.cuda.current_device()
		print(f'ID of current CUDA device: {cuda_id}')
		device = 'cuda:' + str(cuda_id)
		print(f'Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}')
	else:
		print('CUDA not available')
		use_gpu = False


def read_data(file):
	"""Read h5ad object
	"""

	adata = scvi.data.read_h5ad(file)
	return adata

def read_meta(path):
	"""Read metadata file.
	"""
	
	meta = pd.read_csv(path, sep='\t')
	meta = meta[['FGID', 'AGE_SAMPLING', 'SEX', 'BMI', 'EVER_SMOKER', 'Blood donations within 2 years']]
	meta.rename(columns={'FGID':'FINNGENID'}, inplace=True

	return meta

def add_meta(adata, path):
	"""Add metadata to the annData object.
	"""

	meta = read_meta(path)

	id_df = pd.DataFrame(adata.obs[['FINNGENID']])
	id_df.reset_index(inplace=True)
	merged = id_df.merge(meta, how='left', on='FINNGENID')
	merged = merged.set_index('barcode')

	adata.obs['AGE_SAMPLING'] = merged['AGE_SAMPLING']
	adata.obs['SEX'] = pd.Categorical(merged['SEX'])
	adata.obs['EVER_SMOKER'] = merged['EVER_SMOKER']
	adata.obs['Blood donations within 2 years'] = merged['Blood donations within 2 years']


def setup_anndata(adata):
	"""Set up the anndata for scVI.

	Args:
		adata: AnnData object
	"""

	scvi.model.SCVI.setup_anndata(
		adata,
		batch_key='Batch',
		labels_key='seed_labels',
		categorical_covariate_keys=['SEX'],
		continuous_covariate_keys=['AGE_SAMPLING']
	)


def set_scvi_model(adata, params):
	"""Set up the scVi model.
	
	Args:
		adata: AnnData object

	Returns:
		scvi_model
	"""

	scvi_model = scvi.model.SCVI(
		adata,
		n_latent=params['n_latent'],
		n_layers=params['n_layers'],
		n_hidden=params['n_hidden'],
		gene_likelihood='nb',
		encode_covariates=True,
		deeply_inject_covariates=False,
		use_layer_norm='both',
		use_batch_norm='none'
	)
	
	scvi_model.to_device(device)

	return scvi_model


def train_scvi(scvi_model, params, plan_kwargs, multigpu=False):
	"""Train scVI model.
	"""

	if multigpu:
		scvi_model.train(
                max_epochs=params['max_epochs_scVI'],
                plan_kwargs=plan_kwargs,
                early_stopping=True,
                early_stopping_monitor='elbo_validation',
                early_stopping_patience=10,
                early_stopping_min_delta=0.0,
                check_val_every_n_epoch=1,
                accelerator='gpu',
		devices=-1,
		strategy='ddp_find_unused_parameters_true'
        	)
	else:
		scvi_model.train(
			max_epochs=params['max_epochs_scVI'],
			plan_kwargs=plan_kwargs,
			early_stopping=True,
			early_stopping_monitor='elbo_validation',
			early_stopping_patience=10,
			early_stopping_min_delta=0.0,
			check_val_every_n_epoch=1,
			accelerator='gpu',
		)



def train_scvi(scvi_model, params, multigpu=False):
	"""Train scANVI model.

	Args:
		scvi_model
		params

	Returns:
		scanvi_model
	"""

	scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, unlabeled_category='Unknown')

	if multigpu:
  		scanvi_model.train(
                	max_epochs=params['max_epochs_scANVI'],
                	train_size=1.0,
                	accelerator='gpu',
			devices=-1,
			strategy='ddp_find_unused_parameters_true'
        	)
	else:
		scanvi_model.train(
		max_epochs=params['max_epochs_scANVI'],
		train_size=1.0,
		accelerator='gpu'
		)

	return scanvi_model


def predict_missing_cell_types(adata, scanvi_model):
	"""Predict missing cell types and get the latent representation from scANVI model.

	Args:
		adata
		scanvi_model
	"""

	SCANVI_LATENT_KEY = 'X_scANVI'
	SCANVI_PREDICTIONS_KEY = 'C_scANVI'

	adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
	adata.obs[SCANVI_PREDICTIONS_KEY] = scanvi_model.predict(adata)


def cell_type_probabilities(scanvi_model, adata):
	"""Get inferred cell type probabilities for each cell.

	Args:
		scanvi_model: trained scANVI model
		adata: AnnData object
	"""

	df_probs = scanvi_model.predict(adata, soft=True)
	
	for i in df_probs.columns:
		adata.obs['probs_scANVI_' + str(i)] = df_probs[i]


def elbo(model, loss):
	"""Plot elbow plot.
	"""

	elbo = model.history[loss]
	elbo.to_csv()
	fig, axs = plt.subplots(1)
	axs.plot(elbo, label='elbo')
	plt.savefig(f'elbo_{loss}.png', dpi=100)


def save_model(model, dirname):
	"""Save the model in dir 'dirname'
	"""

	model.save(dirname, overwrite=True)


def normalize(adata):
	"""Normalize data for UMAP and freeze the state in adata.raw.
	"""

	adata.layers['counts'] = adata.X.copy()
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)
	adata.raw = adata

def scanvi_umap(adata, SCANVI_LATENT_KEY, SCANVI_PREDICTIONS_KEY):
	"""Calculate and plot UMAP with SCANVI keys
	"""

	sc.pp.neighbors(adata, use_rep=SCANVI_LATENT_KEY)
	sc.tl.umap(adata)

	sc.pl.umap(adata, color=['predicted.celltype.l2', SCANVI_PREDICTIONS_KEY], wspace=0.6, save='_scVI_scANVI_seeds_normalized.png')

