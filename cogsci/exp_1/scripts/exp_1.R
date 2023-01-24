#libraries
library(lsa)
library(vegan)

# set working dir to the dir the script is in
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Load data
leuven_animals = read.csv('../../data/feature_listing/animal_leuven_norms.csv')
leuven_tools = read.csv('../../data/feature_listing/artifacts_leuven_norms.csv')
gpt_summer_no_mc = read.csv('../../data/feature_listing/gpt_response_yes_wide_no_matrix_completion.csv')

# get cosine distance matrix for leuven data 
## combine leuven data into one data frame, the columns are not the same
## for all the columns that are not in both data frames, let the value be 0
leuven = merge(leuven_animals, leuven_tools, by = 'X', all = TRUE)
leuven[is.na(leuven)] = 0

## find a cosine distance matrix for the leuven data
# leuven_cosine_dist_all_feats = 1-  lsa::cosine(t(data.matrix(leuven)))
leuven_cosine_dist_all_feats = as.matrix(dist(data.matrix(leuven)))

leuven_mds <- cmdscale(leuven_cosine_dist_all_feats,k = 3)
leuven_cosine_dist = 1-  lsa::cosine(t(data.matrix(leuven_mds)))
## set the row and column names to the names of the leuven data
rownames(leuven_cosine_dist) = leuven$X
colnames(leuven_cosine_dist) = leuven$X

# find cosine distance matrix for gpt data
# gpt_cosine_dist_summer_no_mc_all_feats = 1-  lsa::cosine(t(data.matrix(gpt_summer_no_mc)))
gpt_cosine_dist_summer_no_mc_all_feats = as.matrix(dist(data.matrix(gpt_summer_no_mc)))
gpt_mds_summer_no_mc <- cmdscale(gpt_cosine_dist_summer_no_mc_all_feats,k = 3)
gpt_cosine_dist_summer_no_mc = 1-  lsa::cosine(t(data.matrix(gpt_mds_summer_no_mc)))
rownames(gpt_cosine_dist_summer_no_mc) = gpt_summer_no_mc$Concept
colnames(gpt_cosine_dist_summer_no_mc) = gpt_summer_no_mc$Concept


# sanity check, proc between leuven_cosine_dist and leuven_cosine_dist_all_feats
leuven_sanity_proc <- protest(leuven_cosine_dist, leuven_cosine_dist_all_feats)
summary(leuven_sanity_proc)
cat(1 - leuven_sanity_proc$ss, leuven_sanity_proc$signif)


# proc between leuven and human
exp_1_proc <- protest(leuven_cosine_dist, gpt_cosine_dist_summer_no_mc)
cat(1 - exp_1_proc$ss, exp_1_proc$signif)

