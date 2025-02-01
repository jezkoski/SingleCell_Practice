install.packages("BiocManager")
BiocManager::install("biomaRt")

library("biomaRt")

ensembl <- useEnsembl(biomart="genes", dataset="hsapiens_gene_ensembl")

filters <- listFilters(ensembl)
attributes <- listAttributes(ensembl)

filters <- filters[1:3,]

bm <- getBM(attribures = c("ensembl_gene_id", "chromosome_name", "start_position", "end_position"), mart=ensembl)
write.table(bm, file="biomart_gene_post.txt")


