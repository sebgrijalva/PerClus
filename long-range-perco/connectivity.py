import numpy as np
import time
import matplotlib.pyplot as plt

from numba import jit,prange

########################################"

import hoshen
import randomsurf

######## CHECK CONNECTIVITY ##########
######################################
@jit(nopython = True)
def three_point(surface,r):
    L = len(surface)
    counts = [0,0,0,0]
    for i in range(L):
        for j in range(L):
            if surface[i,j]:
                if surface[(i+r)%L, j] == surface[i, j] : counts[0] += 1 #horizontal
                if surface[i, (j+r)%L] == surface[i, j] : counts[1] += 1 #vertical
                if surface[(i+r)%L, (j+r)%L] == surface[i, j] : counts[2] += 1 #diagonal
                if surface[(i+r)%L, j] == surface[i, j] == surface[i,(j+r)%L] : counts[3] += 1 #three-point
    return counts

@jit(nopython = True)
def two_point(surface,r):
    L = len(surface)
    counts = 0
    for i in range(L):
        for j in range(L):
            if surface[i,j]:
                if surface[(i+r)%L, j] == surface[i, j]: counts += 1
                if surface[i, (j+r)%L] == surface[i, j]: counts += 1
    return counts

####--- Rectangular Surfaces ---###

@jit(nopython = True)
def rectangular_3pt(surface,r):
    M = surface.shape[0]
    N = surface.shape[1]
    counts = [0,0,0,0]
    for i in range(M):
        for j in range(N):
            if surface[i,j]:
                if surface[(i+r)%M, j] == surface[i, j] : counts[0] += 1 #horizontal
                if surface[i, (j+r)%N] == surface[i, j] : counts[1] += 1 #vertical
                if surface[(i+r)%M, (j+r)%N] == surface[i, j] : counts[2] += 1 #diagonal
                if surface[(i+r)%M, j] == surface[i, j] == surface[i,(j+N)%L] : counts[3] += 1 #three-point
    return counts

@jit(nopython = True)
def rectangular_2pt(surface,r):
    M = surface.shape[0]
    N = surface.shape[1]
    counts = [0,0]
    for i in range(M):
        for j in range(N):
            if surface[i,j]:
                if surface[(i+r)%M, j] == surface[i, j] : counts[0] += 1 #horizontal
                if surface[i, (j+r)%N] == surface[i, j] : counts[1] += 1 #vertical
    return counts

################################################################################

@jit(nopython=True)#, parallel = True)
def two_point_full(surface):
    '''takes all the possible distances'''
    L = len(surface)
    counts = np.zeros(L//2)
    #numba optimizes well this:
    for i in range(L):
        for j in range(L):
            if surface[i,j]:
                #shift the ith row until (i,j) is the first index and count from there
                #remark : numba's prange doesn't understand np.roll
                row = np.roll( (surface[i,:] == surface[i,j])*1, -j )
                col = np.roll( (surface[:,j] == surface[i,j])*1, -i )
                counts += (row[:L//2] + col[:L//2])
    return counts


############### EXPERIMENTAL: FAST COUNTING OF ALL RADIUS ######################
import collections

def list_duplicates(seq):
    duplicates = collections.defaultdict(list) #create dictionary list
    for index,item in enumerate(seq):
        if item: duplicates[item].append(index) #add 'index' to class 'item'
    return (indexes for item,indexes in duplicates.items() if len(indexes) > 1 )

@jit(nopython = True)
def measure_distances(item,L):
    return [[ min(item[j]-item[i], L-(item[j]-item[i])) for j in range(i+1,len(item)) ] for i in range(len(item)-1)]

def two_point2(surface):
    final = collections.Counter()
    for row in surface:
        #generate a list of the indexes of same cluster sites
        duplicates = sorted(list_duplicates(row))

        for item in duplicates:
            measure = measure_distances(np.array(item),len(row))
            flat_list = [x for sublist in measure for x in sublist]
            final += collections.Counter( flat_list )

    return final#list(final.keys()),list(final.values())

###########################################################
##############################################################################


'''
#---Three-Point Connectivity---
alpha = 0.25
height = -0.2240
samples = 10000
sizes = [2**11]

for L in sizes:
    print(f'3pt connectivity: Samples = {samples}, L = {L}, Alpha= {alpha}')
    # Build radius array:
    build = np.sort([2**x for x in range(20)]+[3*2**x for x in range(20)])#+[5*2**x for x in range(20)])
    #build = np.arange(1,L)
    radii = build[build <= L//2]

    #store three point:
    counts = np.zeros((len(radii),4))

    start1 = time.time()
    sp_den = randomsurf.build_spden(L, alpha)
    for _ in range(samples):
        surface = ( randomsurf.gaussian_field(L, alpha, sp_den) > height )*1
        #surface = randomsurf.percolation(L)

        surface = hoshen.get_clusters( surface )
        #print(surface)
        measurement = np.array([three_point(surface, r) for r in radii])

        for i in range(4):
            counts[:,i] += measurement[:,i]/(L**2*samples)

        #del surface #dump memory

    print(f'Total elapsed time = {round(time.time() - start1, 3)}')

    result = counts[:,3] #three-point only


    np.savetxt(f'data/three_point/L={L}-alpha={alpha}-s={samples}.txt',result, fmt='%1.4f,',)

    plt.semilogx(radii/L,radii**(6/4*5/24)*result,'.',ls = "--")

plt.show()
'''

##################################
############################
#----Ratio with 2pt----
#######################################
#######################################
'''
alpha = 0.0
height = -0.2346
samples = 10000
sizes = [2**9,2**10]

for L in sizes:
    print(f'Ratio 3pt/2pt: Samples = {samples}, L = {L}, Alpha= {alpha}')
    # Build radius array:
    build = np.sort([2**x for x in range(20)]+[3*2**x for x in range(20)])
    #build = np.arange(1,L)
    radii = build[build <= L//2]

    #store three point:
    counts = np.zeros((len(radii),4))

    start1 = time.time()
    sp_den = randomsurf.build_spden(L, alpha)
    for _ in range(samples):
        surface = ( randomsurf.gaussian_field(L, alpha, sp_den) > height )*1
        #surface = randomsurf.percolation(L)

        surface = hoshen.get_clusters( surface )
        #print(surface)
        measurement = np.array([three_point(surface, r) for r in radii])

        for i in range(4):
            counts[:,i] += measurement[:,i]/(L**2*samples)

        #del surface #dump memory

    print(f'Total elapsed time = {round(time.time() - start1, 3)}')


    ratio = counts[:,3]/np.sqrt(counts[:,0]*counts[:,1]*counts[:,2])

    np.savetxt(f'data/ratio/L={L}-alpha={alpha}-s={samples}.txt',ratio, fmt='%1.4f,',)
    plt.plot(np.log2(radii/L),ratio,'.',ls = "--")


plt.show()
'''

##################

###################################################
#----------TWO-POINT CONNECTIVITY---
###################################################
'''
alphas = [2/3., 0.75]
heights = [-0.2033, -0.1989]
sample_space = [10000]*5
sizes = [512]*5

l=0
for alpha in alphas:
    L = sizes[l]
    h = heights[l]
    samples = sample_space[l]
    print(f'2pt Connectivity: Samples = {samples}, L = {L}, Alpha= {alpha}')
    # Build radius array:
    build = np.sort([2**x for x in range(20)]+[3*2**x for x in range(20)])
    radius = build[build <= L//2]

    #radius = np.arange(1,28)
    #radius = radius[ radius <= L//2 ]

    #two_point:
    result = np.zeros( (samples,len(radius)) )
    #result = 0

    sp_den = randomsurf.build_spden(L,alpha)
    start1 = time.time()
    for s in range(samples):
        #start=time.time()
        surface = ( randomsurf.gaussian_field(L, alpha,sp_den) > h )*1
        #surface = randomsurf.percolation(L)
        #surface = randomsurf.berry(L)

        surface = hoshen.get_clusters( surface )
        #result += np.array(two_point_full(surface))

        result[s,:] = np.array([two_point(surface, r) for r in radius])/(2*L**2)

        del surface #dump memory
    print(f'Total elapsed time = {round(time.time() - start1, 3)}')
    twopt = [np.mean(result[:,r]) for r in range(len(radius))]
    error = [np.std(result[:,r])/np.sqrt(samples) for r in range(len(radius))]

    np.savetxt(f'data/two_point/L={L}-alpha={alpha}-s={samples}.txt',(radius,twopt,error), fmt='%1.4f',)
    #np.savetxt(f'data/two_point/L={L}-Percolation-s={samples}.txt',(radius,twopt,error), fmt='%1.4f',)
    plt.errorbar(
        x = radius,
        y = twopt,
        #y = twopt,
        yerr = error,
        marker = '.',
        #color = 'k',
        capsize = 2,
        label = f'alpha = {alpha}'
    )
    plt.legend()
    l+=1

plt.show()
'''

# Rectangular Lattice
'''
alpha = 2/3.
height = -0.2345
samples = [100000,100000,20000,10000]

sizes = [64,128,256,512]
factor = 1

l=0
for L in sizes:

    print(f'2pt Connectivity: Samples = {samples[l]}, Size = {factor*L}x{L}, Alpha= {alpha}')
    # Build radius array:
    build = np.sort([2**x for x in range(20)]+[3*2**x for x in range(20)])
    radius = build[build <= L//2]
    #radius = radius[radius > L//20]
    result = np.zeros( (2,samples[l],len(radius)) )

    sp_den = randomsurf.rectangular_spden(factor*L,L,alpha)
    start1 = time.time()

    for s in range(samples[l]):
        #start=time.time()
        surface = ( randomsurf.rectangular_gaussian_field(factor*L,L, alpha,sp_den) > height )*1
        #surface = randomsurf.percolation(L)
        #surface = randomsurf.berry(L)

        surface = hoshen.rectangular_get_clusters( surface )

        i = 0
        for r in radius:
            counts = np.array(rectangular_2pt(surface, r))/(factor*L*L)
            result[0,s,i] = counts[0]
            result[1,s,i] = counts[1]
            i += 1
        del surface #dump memory


    print(f'Total elapsed time = {round(time.time() - start1, 3)}')
    twopt = [np.mean(result[1,:,k]) for k in range(len(radius))]
    error = [2*np.std(result[1,:,k])/np.sqrt(samples[l]) for k in range(len(radius))]


    np.savetxt(f'data/two_point/L={L}-alpha={alpha}-s={samples}.txt',(twopt,error), fmt='%1.4f,',)
    #np.savetxt(f'data/two_point/L={L}-Percolation-s={samples}.txt',result, fmt='%1.4f,',)
    plt.errorbar(
        x = radius/L,
        y = radius**(5/24)*twopt,
        #y = twopt,
        yerr = error,
        marker = '.',
        #color = 'k',
        capsize = 2
    )
    l+=1

plt.show()
'''
##########################

# -- Critical height by minimizing the distance btw 2pt connectivities at different L
'''
# Square Lattice:

start=time.time()

alpha = 0.25
levels = np.linspace(-0.25,-0.18,16)
samples = 200
exponent = [5/24.]#np.linspace(0.15,0.3,10)

for exp in exponent:

    plt.figure()
    for L in [16,64,256]:  ##TODO: P12(2L) / P12(L) = f(r/L)

        radius = np.int(L//2) #choose a particular radius to measure
        print(f'Samples = {samples}, Size = {L}x{L}, Alpha= {alpha}, Measuring at = {radius}')

        result = np.zeros((samples,len(levels)))
        sp_den = randomsurf.build_spden(L,alpha)

        for s in range(samples):

            surface = randomsurf.gaussian_field(L,alpha,sp_den)

            i = 0
            for level in levels:

                #surface = randomsurf.percolation(L, p = level)
                #surf = hoshen.get_clusters( surface )
                surf = hoshen.get_clusters( (surface > level)*1 )
                counts = np.array(two_point(surf,radius))/(2*L**2)
                result[s,i] = counts*radius**exp
                i += 1

        connec = [np.mean(result[:,i]) for i in range(len(levels))]
        error = [np.std(result[:,i])/np.sqrt(samples) for i in range(len(levels))]

        #np.savetxt(f'data/connectivity/L={L}-alpha={alpha}-s={samples}.txt',(levels,result), fmt='%1.4f,',)
        plt.title(f'Square Lattice, Two-Point Connectivity, alpha= {alpha}, exponent = {exp}')
        plt.errorbar(
            x = levels,
            y = connec,
            yerr = error,
            marker = '.',
            capsize = 2,
            label = f'L = {L}'
        )
        plt.legend()

end=time.time()
print(f'Elapsed time = {round(end-start,4)}')

plt.show()
'''
# Rectangular Lattice:
'''
start=time.time()

alpha = 2/3
levels = np.linspace(-0.25,-0.15,18)
samples = 10000
factor = 2

for L in [2**5,2**6,2**7]:

    radius = np.int(L//2) #choose a particular radius to measure
    print(f'Samples = {samples}, Size = {factor*L}x{L}, Alpha= {alpha}, Measuring at = {radius}')

    result = np.zeros((2,samples,len(levels)))
    sp_den = randomsurf.rectangular_spden(factor*L,L,alpha)

    for s in range(samples):

        surface = randomsurf.rectangular_gaussian_field(factor*L,L,alpha,sp_den)

        i = 0
        for level in levels:
            #surface = randomsurf.percolation(L, p = level)
            #surf = hoshen.get_clusters( surface )
            surf = hoshen.rectangular_get_clusters( (surface > level)*1 )
            counts = np.array(rectangular_2pt(surf,radius))/(factor*L*L)
            result[0,s,i] = counts[0]*radius**(0.213) # horizontal
            result[1,s,i] = counts[1]*radius**(0.213) # vertical
            i += 1

    connec = [np.mean(result[0,:,i]) for i in range(len(levels))]
    error = [np.std(result[0,:,i])/np.sqrt(samples) for i in range(len(levels))]

    #np.savetxt(f'data/connectivity/L={L}-alpha={alpha}-s={samples}.txt',(levels,result), fmt='%1.4f,',)
    plt.title(f'Rectangular Lattice (M/N = {factor}), Two-Point Connectivity, alpha= {alpha}')
    plt.errorbar(
        x = levels,
        y = connec,
        yerr = error,
        marker = '.',
        capsize = 2,
        label = f'L = {L}'
    )
    plt.legend()

end=time.time()
print(f'Elapsed time = {round(end-start,4)}')

plt.show()
'''

# -- Convergence of different system sizes at the critical level:
'''
# Square Lattice:
start=time.time()

alpha = 0.25
sizes = np.linspace(8,3000,20,dtype = np.int16)
samples = 100

result = np.zeros((samples,len(sizes)))

i = 0

for L in sizes:

    print(f'Samples = {samples}, Size = {L}x{L}, Percolation')
    radius = L//2

    #sp_den = randomsurf.build_spden(L,alpha)
    for s in range(samples):
        #surface = randomsurf.gaussian_field(L,alpha,sp_den)

        surface = randomsurf.percolation(L, p = 0.5927)
        surf = hoshen.get_clusters( surface )
        #surf = hoshen.get_clusters( (surface > level)*1 )
        counts = np.array(two_point(surf,radius))/(2*L**2)
        result[s,i] = radius**(5/24)*counts

    connec = [np.mean(result[:,i]) for i in range(len(sizes))]
    error = [np.std(result[:,i])/np.sqrt(samples) for i in range(len(sizes))]
    i += 1
#np.savetxt(f'data/connectivity/L={L}-alpha={alpha}-s={samples}.txt',(levels,result), fmt='%1.4f,',)
plt.legend()
plt.title(f'Square Lattice, Two-Point Connectivity, Percolation')
plt.xlabel('System Size')
plt.ylabel('Connectivity at L/2')
plt.errorbar(
    x = sizes,
    y = connec,
    yerr = error,
    marker = '.',
    capsize = 2
)

end=time.time()
print(f'Elapsed time = {round(end-start,4)}')

plt.show()
'''

# -- Convergence of different samples at the critical level to a given value of connectivity (independent of size):
'''
# Square Lattice:
start=time.time()

alpha = 0.4
sizes = [8,16,32,64,128,256,512]
samples = 10000


for L in sizes:
    result = np.zeros(samples)
    count = []
    error = np.zeros(samples)

    print(f'Samples = {samples}, Size = {L}x{L}, alpha = {alpha}')
    radius = L//2

    sp_den = randomsurf.build_spden(L,alpha)

    for s in range(samples):

        surface = randomsurf.gaussian_field(L,alpha,sp_den)
        #surface = randomsurf.percolation(L, p = 0.5927)

        #surf = hoshen.get_clusters( surface )
        surf = hoshen.get_clusters( (surface > -0.2172)*1 )

        count.append( radius**(5/24)*two_point(surf,radius)/(2*L**2) )

        result[s] = np.mean(count)
        error[s] = np.std(count)/np.sqrt(s+1)

        #connec = [np.mean(result[:,i]) for i in range(len(sizes))]
        #error = [np.std(result[:,i])/np.sqrt(samples) for i in range(len(sizes))]

    #np.savetxt(f'data/connectivity/L={L}-alpha={alpha}-s={samples}.txt',(levels,result), fmt='%1.4f,',)
    plt.title(f'Square Lattice, Two-Point Connectivity, Alpha = {alpha}')
    plt.xlabel('Total Samples')
    plt.ylabel('Connectivity at L/2')
    plt.errorbar(
        x = range(samples),
        y = result,
        yerr = error,
        marker = '.',
        ls = ':',
        capsize = 0,
        label = f'L = {L}'
    )
    plt.legend()

end=time.time()
print(f'Elapsed time = {round(end-start,4)}')

plt.show()
'''
#-- Rectangular Lattice
'''
start=time.time()

alpha = 0.25
parameters = np.linspace(-.24,-0.21,8)
#parameters = np.linspace(.58,.61,20)
samples = [80000,30000,20000,10000,5000]

factor = 3
l = 0
for L in [16,32,64,128,256]:

    radius = np.int(L//2) #choose a particular radius to measure
    print(f'Samples = {samples[l]}, Size = {L}x{factor*L}, Alpha= {alpha}, Measuring at = {radius}')

    result = np.zeros((2,samples[l],len(parameters)))
    sp_den = randomsurf.rectangular_spden(L,factor*L,alpha)

    i = 0
    for parameter in parameters:
        for s in range(samples[l]):
            surface = 1*(randomsurf.rectangular_gaussian_field(L,factor*L,alpha,sp_den) > parameter)
            #surface = randomsurf.rectangular_percolation(factor*L, L, p = parameter)

            #surf = hoshen.rectangular_get_clusters( (surface > level)*1 )
            surf = hoshen.rectangular_get_clusters( surface )
            counts = np.array(rectangular_2pt(surf,radius))/(L*factor*L)
            result[0,s,i] = radius**(5/24)*counts[0]
            result[1,s,i] = radius**(5/24)*counts[1]
        i += 1

    connec = [np.mean(result[0,:,i]) for i in range(len(parameters))]
    error = [np.std(result[0,:,i])/np.sqrt(samples[l]) for i in range(len(parameters))]

    #np.savetxt(f'data/connectivity/(rectangular)-L={L}-factor={factor}-alpha={alpha}-s={samples}.txt',(levels,result[0],result[1]), fmt='%1.4f,',)
    plt.figure(0)
    plt.title(f'Rectangular Lattice, M/N = {factor}, Two-Point Connectivity, alpha= {alpha}')
    #plt.title(f'Rectangular Lattice, M/N = {factor}, Two-Point Connectivity, Percolation')
    plt.errorbar(
        x = parameters,
        y = connec,
        yerr = error,
        marker = '.',
        capsize = 2
    )
    l+=1

end=time.time()
print(f'Elapsed time = {round(end-start,4)}')

plt.show()
'''

#########################################
#Time Tests

'''

import time
import matplotlib.pyplot as plt
#

for k in range(1):
    start=time.time()
    L=2**10
    #surface=hoshen.get_clusters(randomsurf.create(L,-.234614))
    surface=hoshen.get_clusters((randomsurf.gauss_populate(L,0.25)>-.223029).astype(int))
    #plt.matshow(surface)
    #plt.show()
    #
    build=np.logspace(0, 12, num=13, base=2).astype(int)
    build=np.sort(np.concatenate((build,3*build)))
    radius=build[build<=L//2]

    #
    #connec=two_point_full(surface)
    connec=[]
    for r in radius:
        connec.append(r**(5/24)*two_point_parallel(surface,r))
    data=np.array(connec)/L**2

    end=time.time()
    plt.semilogx(radius/L,data,'.')
    print("Elapsed time = ", end-start)
    #plt.plot(range(L//2+1),connec/L**2,'.')

    plt.show()
'''


#########################################################################
#  RECTANGULAR TWO-POINT FUNCTION #
#########################################################################
'''
#---Rectangular Connectivity---

alpha = 0.0
height = -0.23418
samples = 1000
sizes = [2**11]
factor = 2

for L in sizes:
    print(f'horizontal vs vertical connectivity: Samples = {samples}, L = {L}, Alpha= {alpha}')
    # Build radius array:
    build = np.sort([2**x for x in range(20)]+[3*2**x for x in range(20)])#+[5*2**x for x in range(20)])
    #build = np.arange(1,L)
    radii = build[build <= L//2]

    #store three point:
    counts = np.zeros((len(radii),2))

    start1 = time.time()
    #sp_den = randomsurf.rectangular_spden(factor*L, L, alpha)
    for _ in range(samples):
        #surface = ( randomsurf.rectangular_gaussian_field(factor*L, L, alpha, sp_den) > height )*1
        surface = randomsurf.rectangular_percolation(factor*L,L)

        surface = hoshen.rectangular_get_clusters( surface )
        #print(surface)
        measurement = np.array([rectangular_2pt(surface, r) for r in radii])

        for i in range(2):
            counts[:,i] += measurement[:,i]/(L*factor*L*samples)

        #del surface #dump memory

    print(f'Total elapsed time = {round(time.time() - start1, 3)}')

    horizontal = counts[:,0]
    vertical = counts[:,1]

    np.savetxt(f'data/h_vs_v/L={L}-(percolation)-s={samples}.txt',(horizontal,vertical), fmt='%1.5f,',)
    #np.savetxt(f'data/h_vs_v/L={L}-alpha={alpha}-s={samples}.txt',(horizontal,vertical), fmt='%1.5f,',)

    plt.figure(1)
    plt.semilogx(radii/L,radii**(5/24)*horizontal,'.',ls = "--")
    plt.semilogx(radii/L,radii**(5/24)*vertical,'.',ls = "--")
    plt.figure(2)
    plt.semilogx(radii/L,radii**(5/24)*(vertical-horizontal),'.',ls = "--")

plt.show()
'''

################################################################
#################################################################
################BASIC TESTS###################
'''
levels = np.linspace(-.3,-0.1,4)
M = 1024
N = 8*M
alpha = 0.0

sp_den = randomsurf.rectangular_spden(M,N,alpha)

for level in levels:
    surf=(randomsurf.rectangular_gaussian_field(M,N,alpha,sp_den)>level)*1
    #plt.matshow(surf)
    plt.matshow(hoshen.rectangular_get_clusters(surf))
plt.show()
'''
