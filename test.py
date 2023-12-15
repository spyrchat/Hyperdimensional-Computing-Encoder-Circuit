import numpy as np

D_b=4
final_HDC_centroid = np.arange(-100, 21)
final_HDC_centroid_q = np.zeros(len(final_HDC_centroid))

max_centroid = np.max(np.abs(final_HDC_centroid))
min_centroid = np.min(np.abs(final_HDC_centroid))

    #if (final_HDC_centroid[x] >= 0): fact = (2**(D_b-1)-1)/max_centroid
    #else: fact = (2**(D_b-1))/max_centroid
final_HDC_centroid_q = np.round(final_HDC_centroid*(2**(D_b-1)-1)/max_centroid)


'''
for x in range(len(final_HDC_centroid)):
    if (final_HDC_centroid[x] >= 0): final_HDC_centroid_q[x] = (final_HDC_centroid[x]-min_centroid)*((2**(D_b-1)-1)/(max_centroid-min_centroid))
    else: final_HDC_centroid_q[x] = -(max_centroid - final_HDC_centroid[x])*((2**(D_b-1))/(max_centroid-min_centroid))
'''
print(final_HDC_centroid_q)
