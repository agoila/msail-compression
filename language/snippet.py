import random 
import os
import glob
import bisect

with open('pride.txt') as f:
    text = f.read()

L = 4026
N = 256 
sorted_is = []
for n in range(N):
    while True:
        i = random.randrange(len(text)-L+1)
        j_lo = bisect.bisect_left(sorted_is, i)
        if j_lo and abs(sorted_is[j_lo-1]-i)<L: continue 
        if j_lo!=len(sorted_is) and abs(sorted_is[j_lo]-i)<L: continue
        break
    sorted_is.append(i) 
print('Successfully sampled text!')

for i in sorted_is:
    with open('pride_%07d' % i, 'w') as f:
        f.write(text[i:i+L])
os.system('bzip2 pride_*')

total_size = sum(os.path.getsize(f) for 
                 f in glob.glob('*.bz2')) 
print('Compression Ratio: %.2f' % (N*L/float(total_size)))

os.system('rm *.bz2')
