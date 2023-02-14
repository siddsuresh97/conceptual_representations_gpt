#   ========================================
#   Matrix completion on the Leuven data set
#   ========================================

#   This python script 
#     - randomly drops some entries from the Leuven matrix
#     - attempts to recover the dropped entries with the fancyimpute python package
#     - reports some basic performance statistics
#     - plots the recovered matrix against the ground truth
#
#   For questions email <greg.hensleman@gmail.com>
#   Last updated 2023-02-02

from fancyimpute import SoftImpute, BiScaler, IterativeSVD
import numpy as np
import csv
import pandas as pd
import copy
import matplotlib.pyplot as plt

#   Load the Leuven matrix
#   ----------------------

filename="leuven_data_set_numbers_only.mat"

from scipy.io import loadmat
leuven_array = loadmat(filename)
leuven_array = leuven_array['leuven_array']
leuven_array = leuven_array.astype(np.float)

(d1, d2) = leuven_array.shape

dropped_array = copy.deepcopy(leuven_array)
dropped_positive_count = 0
dropped_negative_count = 0

#   Choose dropout parameters
#   -------------------------

p_droptruepos   =   0.10 # probability of dropping a true positive value
p_droptrueneg   =   0.45 # probability of dropping a true negative (zero) value

#   Replace some entries with NaN's
#   -------------------------------

for row in range(d1):
    for col in range(d2):
        if dropped_array[row][col] == 1:
            if np.random.uniform(0,1) < p_droptruepos:
                dropped_array[row][col] = np.NaN
                dropped_positive_count +=1
        else:
            if np.random.uniform(0, 1) < p_droptrueneg:
                dropped_array[row][col] = np.NaN
                dropped_negative_count +=1                

#   Matrix completion with IterativeSVD
#   -----------------------------------
#       -   we apply matrix completion to the partially masked matrix; the result has real-valued entries, which we round to 0/1
#       -   you can play with the rank parameter to get different results
#       -   as Tim predicted, higher ranks seem to support recovery of rows that have a higher degree of variation
#       -   the fancyimpute library has a range of other algorithms, in addition to IterativeSVD

dropped_array_partfilled = copy.deepcopy(dropped_array)
for iter in range(1):
    X_filled = IterativeSVD(min_value=0, max_value=1,rank=20).fit_transform(dropped_array_partfilled)
    X_guessed = np.round(X_filled)
    np.clip(X_guessed, 0, 1)

    for row in range(d1):
        for col in range(d2):
            if X_guessed[row][col] == 1:
                dropped_array_partfilled[row][col] = 1

#   Evaluate performance of the reconstruction procedure
#   ----------------------------------------------------

err_count = np.sum( np.round(X_guessed) != leuven_array )
pct_error = err_count / np.prod(leuven_array.shape)
pct_nonzero_true = np.count_nonzero(leuven_array) / np.product(leuven_array.shape)
pct_pos_correctly_identified = np.count_nonzero( (X_guessed == 1) & (leuven_array ==1) )/np.count_nonzero(leuven_array ==1)
pct_pos_previously_identified = np.count_nonzero(dropped_array == 1)/np.count_nonzero(leuven_array)
err_weighted = np.sum( np.abs( np.round(X_guessed) - leuven_array) )

err_count_above = np.sum( np.abs( np.round(X_guessed) > leuven_array) )
err_count_below = np.sum( np.abs( np.round(X_guessed) < leuven_array) )
# 10258.0
# 10395.0
# 10398

print("Let K    = number of entries in the matrix")
print("Let Kt   = number of positive entries in the matrix (ground truth)")
print("Let Kf   = number of negative entries in the matrix (ground truth)")
print("------------------------------------------------------------------")
print(f"masked positives                            {dropped_positive_count}        ")
print(f"masked negatives                            {dropped_negative_count}        ")
print(f"entries guessed incorrect (num)             {err_count}                     ")
print(f"entries guessed incorrect (num/K)           {pct_error}                     ")
print(f"Kt/K                                        {pct_nonzero_true}              ")
print(f"(# correct pos guess)/Kt                    {pct_pos_correctly_identified}  ")
print(f"(# correct pos not masked)/Kt               {pct_pos_previously_identified} ")
print(f"false positives                             {err_count_above}               ")
print(f"false negatives                             {err_count_below}               ")

# EXAMPLE OUTPUT:
# Let K    = number of entries in the matrix
# Let Kt   = number of positive entries in the matrix (ground truth)
# Let Kf   = number of negative entries in the matrix (ground truth)
# ------------------------------------------------------------------
# masked positives                            3660        
# masked negatives                            384155        
# entries guessed incorrect (num)             5430                     
# entries guessed incorrect (num/K)           0.006088187906369724                     
# Kt/K                                        0.042337012033981734              
# (# correct pos guess)/Kt                    0.9819650423728814  
# (# correct pos not masked)/Kt               0.903072033898305 
# false positives                             4749               
# false negatives                             681  


fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1 )
ax0.imshow( X_guessed, cmap='tab20_r', interpolation='nearest')
ax1.imshow( leuven_array, cmap='tab20_r', interpolation='nearest')
ax0.title.set_text('leuven (reconstructed)')
ax1.title.set_text('leuven')

fig.show()
