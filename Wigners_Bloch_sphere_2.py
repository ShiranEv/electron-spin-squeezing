#%%
"""
Created on Sun May 19 21:01:37 2024

@author: shiranev
"""

import matplotlib
from math import *
from cmath import exp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from qutip import *
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, MultipleLocator
from qutip import *
from cycler import cycler
#%%
def wigner_sphere(state, title="", steps=300, view_1=30, view_2=45, x_len=7, y_len=7, save=None):
    theta = np.linspace(0, np.pi, steps)
    phi = np.linspace(0, 2 * np.pi, steps)
    W_spin, THETA, PHI = spin_wigner(state, theta, phi)
    fig = plt.figure(dpi=300, figsize=(x_len, y_len))
    ax = fig.add_subplot(111, projection='3d', computed_zorder="False")
    ax.view_init(view_1, view_2)

    wigner = np.transpose(W_spin)
    minmax = 1.1
    ax.set_xlim(-minmax, minmax)
    ax.set_ylim(-minmax, minmax)
    ax.set_zlim(-minmax, minmax * 1.2)  # Adjust z-limit to provide more space
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    plt.axis("off")
    ax.zaxis.labelpad = -1
    steps = len(wigner)
    bbox = ax.get_position()
    new_bbox = (bbox.x0, bbox.y0 + 0.2, bbox.width, bbox.height)
    ax.set_position(new_bbox)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones(steps))
    wigner = np.real(wigner)
    wigner_max = np.real(np.amax(np.abs(wigner)))
    wigner_c1 = cm.seismic((wigner + wigner_max) / (2 * wigner_max))

    ax.plot_surface(x, y, z, facecolors=wigner_c1, vmin=-wigner_max,
                    vmax=wigner_max, rcount=steps, ccount=steps, linewidth=0,
                    zorder=0.1, antialiased=None, alpha=1)

    lw = 1.5
    l = 0.7
    ax.quiver(0, 0, 1, 0, 0, l, color='k', alpha=1, lw=lw, zorder=20)
    ax.quiver(1, 0, 0, l, 0, 0, color='k', alpha=1, lw=lw, zorder=20)
    ax.quiver(0, 1, 0, 0, l, 0, color='k', alpha=1, lw=lw, zorder=20)

    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()

    #%%
def coherent_t_p(thet, ph,N):
    N_k = N
    return sum([sqrt(np.math.comb(N_k,k))*(cos(thet/2)**(N_k-k))*((np.exp(1j*ph)*sin(thet/2))**k)*basis(N+1,k) for k in range(N+1)]).unit()

#%%
N=20
g = 2.0
sigma= 2
chi = 0.5
s=N/2
S_Z = jmat(s, 'z')
S_X=jmat(s,'x')
S_Y=jmat(s,'y')

ph_x=np.pi/2
ph_y=0
ph_z=0.3
Rx=(1j*ph_x*S_X).expm()
Ry=(1j*ph_y*S_Y).expm()
Rz=(1j*ph_z*S_Z).expm()

q = g*N/2
operator = ((-1/(4*sigma**2)) * (q*qeye(N+1) - g * (S_Z + (N/2) * qeye(N+1)))**2).expm()
#operator = (-1j * chi/2* S_Z * S_Z).expm()
coherent_state = operator* spin_coherent(N/2, np.pi/2,0)

state = coherent_state.unit()
wigner_sphere(state, title="measurement", steps=100, view_1=11,view_2=18,save = 'figs/int_bloch_0_5.svg')


#%%
theta=0.5

squeezed = (1j*theta*S_Z**2).expm() * spin_coherent(N/2, np.pi/2,0)
wigner_sphere(squeezed.unit(), title="squeezing", steps=80, view_1=0,view_2=20)


# %%
def plot_wigner_sphere(state, plot_name=None):
    steps = 80
    x_len = 4
    y_len = 4
    view_1 = 10
    view_2 = 20
    theta = np.linspace(0, np.pi, steps)
    phi = np.linspace(0, 2*np.pi, steps*2)
    W_spin, THETA, PHI = spin_wigner(state, theta, phi)
    w = np.transpose(W_spin)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones(steps*2))
    wigner_max = np.amax(np.abs(w))
    fig = plt.figure(dpi=80, figsize=(x_len, y_len))
    ax = fig.add_subplot(111, projection='3d', computed_zorder="False")
    ax.view_init(view_1, view_2)
    plt.axis("off")
    wigner_c1 = cm.seismic((w + wigner_max) / (2 * wigner_max))
    ax.plot_surface(x, y, z, facecolors=wigner_c1, vmin=-wigner_max,
                    vmax=wigner_max, rcount=steps, ccount=steps*2, linewidth=0,
                    antialiased=False, alpha=1)
    lw = 1.5
    l = 0.7
    ax.quiver(0, 0, 1, 0, 0, l, color='k', alpha=1, lw=lw, zorder=20)
    ax.quiver(1, 0, 0, l, 0, 0, color='k', alpha=1, lw=lw, zorder=20)
    ax.quiver(0, 1, 0, 0, l, 0, color='k', alpha=1, lw=lw, zorder=20)
    if plot_name is not None:
        plt.savefig(plot_name)  
    plt.show()

# %% figure 1

N=20
s=N/2
state = spin_coherent(s, 0,0)
plot_wigner_sphere(state, 'figs/initial.svg')

state = spin_coherent(s, np.pi/2,0)
plot_wigner_sphere(state, 'figs/after_bs.svg')

g = 2.0
q = g*s
sigma= 2
operator = ((-1/(4*sigma**2)) * (q*qeye(N+1) - g * (jmat(s, 'z') + (N/2) * qeye(N+1)))**2).expm()

state_sq = (operator * state).unit()
plot_wigner_sphere(state_sq, 'figs/squeeze.svg')

BS_op = (-1j*(np.pi/2)*jmat(s, 'x')).expm()
state_sq = BS_op * state_sq
plot_wigner_sphere(state_sq, 'figs/after_sec_bs.svg')

ph_z=np.pi/4
phase_op = (-1j*(ph_z/2)*jmat(s, 'z')).expm()
state_sq = phase_op * state_sq
plot_wigner_sphere(state_sq, 'figs/squeeze_after_sample.svg')

state_sq = BS_op * state_sq
plot_wigner_sphere(state_sq, 'figs/squeeze_before_measure.svg')

state = phase_op * state
plot_wigner_sphere(state, 'figs/after_sample.svg')

state = BS_op * state
plot_wigner_sphere(state, 'figs/before_measure.svg')



# %% figure 2
s=50
c=np.pi
S_z = jmat(s, 'z')
state = (-1j * (c / 2) * S_z ** 2).expm() * spin_coherent(s, np.pi / 2, 0)
plot_wigner_sphere(state)
delta = 0.5 * np.arctan(4 * np.sin(c / 2) * np.cos(c / 2) ** (s*2 - 2) / (1 - np.cos(c) ** (s*2 - 2)))
state_rot = (1j * (np.pi / 2 - delta) * jmat(s, 'x')).expm() * state
plot_wigner_sphere(state_rot)
# plot_wigner_sphere(OAT * spin_coherent(s, np.pi/2,0), 'figs/chi={}.svg'.format(0.26))



# %% 
state = (basis(21,0)+basis(21,20)).unit()
plot_wigner_sphere(state)
delta = 0.5 * np.arctan(4 * np.sin(c / 2) * np.cos(c / 2) ** (s*2 - 2) / (1 - np.cos(c) ** (s*2 - 2)))
state_rot = (1j * (np.pi / 2 - delta) * jmat(s, 'x')).expm() * state
plot_wigner_sphere(state_rot)
F_y = 4 * variance(jmat(s, 'y'), state)
F_z = 4 * variance(S_z, state_rot)
print("Fisher y = {}".format(F_y))
print("Fisher z = {}".format(F_z)) 
print(4*s*(s+1)/2)
print(4*s*(s+1)/2 - F_y)
print(4*s*(s+1)/2 - F_z)

# %% figure 3
s=3
n=2*s

sigma = 1/4
c=1/(2*sigma**2)    
c=3
h = s
operator = ((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm()
state = (operator * spin_coherent(s, np.pi/2,0)).unit()
plot_wigner_sphere(state)
plot_fock_distribution(state)

# %%
N=20
s=N/2
c=0.1
h = s
plt.plot([comb(N,n) for n in range(N+1)], '.')
plt.plot([comb(N,n)*np.exp(-(c/2)*(n-h)**2) for n in range(N+1)], '.')
plt.plot([comb(N,n)*np.exp(-(2/2)*(n-h)**2) for n in range(N+1)], '.')
plt.plot([comb(N,n)*np.exp(-(3/2)*(n-h)**2) for n in range(N+1)], '.')

#%%
fisher = 4 * variance(jmat(s, 'y'), state)
print('Fisher = {}'.format(fisher))
phase_f = 1/np.sqrt(fisher)


avg_sx = expect(jmat(s, 'x'), state)
print('avg sx = {}'.format(avg_sx))
sq_par_square = 4 * variance(jmat(s, 'z'), state) / avg_sx**2
print('squeezing parameter = {}'.format(sq_par_square))
sq_phase = np.sqrt(sq_par_square/(2*s))
print('Fisher phase = {}'.format(phase_f))
print('squeezing phase = {}'.format(sq_phase))

# %%
# %% figure 3
N=50
s=N/2
# chi = 0.1
# sigma = 1/np.sqrt(2*chi)
sigma = 3
chi = 1/(2*sigma**2)
print(chi)
h = s-3 #jmat is opposite - positive then negative
operator = ((-1/(4*sigma**2)) * (h*qeye(N+1) - (-jmat(N/2, 'z') + (N/2) * qeye(N+1)))**2).expm()
operator = ((-chi/2) * (h*qeye(N+1) - (jmat(N/2, 'z') - (N/2) * qeye(N+1)))**2).expm()
operator.norm()
plot_wigner_sphere((operator * spin_coherent(N/2, np.pi/2,0)).unit(), 'figs/sigma={}.svg'.format(sigma))

# %%
N=20
s=N/2
sigma = 40
chi = 1/(2*sigma**2)
print('sigma='+str(sigma))
print('chi='+str(chi))
print('sqrt(s/2='+str(np.sqrt(s/2)))
h=s
operator = ((-1/(4*sigma**2)) * (-jmat(s, 'z') + (s-h) * qeye(N+1))**2).expm()
state = operator * spin_coherent(s, np.pi/2,0)
# plot_wigner_sphere(state)
state = state.unit()
print('initial var='+str(s/2))
print('var='+str(variance(jmat(s, 'z'), state)))
new_sigma_square = 1/((1/((s/2))) + (1/(sigma**2)))
print(new_sigma_square)
chi_2 = 1/(2*(s/2))
print('sigma^2='+str(sigma**2))
print(1/(2*(chi+chi_2)))
plot_fock_distribution(state)

# %%
plot_wigner_sphere(state)
# %%

print(sigma**2)
gaussian = sum([np.exp((-1/(4*(s/2))) * (n-h)**2) * np.exp((-1/(4*sigma**2)) * (n-h)**2) * basis(N+1,n) for n in range(N+1)]).unit()
print('var='+str(variance(jmat(s, 'z'), gaussian)))
new_sigma_square = 1/((1/((s/2))) + (1/(sigma**2)))
plot_fock_distribution(gaussian)
new_state = sum([np.exp((-1/(4*new_sigma_square)) * (n-h)**2) * basis(N+1,n) for n in range(N+1)]).unit()
plot_fock_distribution(new_state)
print('var='+str(variance(jmat(s, 'z'), new_state)))

# %%
N=15
s=N/2
chi_2 = 1/(2*(s/2)**2)
print('var='+str(variance(jmat(s, 'z'), spin_coherent(s, np.pi/2,0))))
print(s/2)
# %%
N_arr=np.arange(10, 60, 10)
chi = np.linspace(0.1, 5, 20)
calc_fisher = np.zeros((len(N_arr), len(chi)))
fisher = np.zeros((len(N_arr), len(chi)))
fisher_eithan = np.zeros((len(N_arr), len(chi)))
for j,N in enumerate(N_arr):
    h_range = np.linspace(0, N, 5*N+1)
    delta_h = h_range[1] - h_range[0]
    s=N/2
    for i,c in enumerate(chi):
        for h in h_range:
            operator = ((-c/2) * (-jmat(s, 'z') + (s-h) * qeye(N+1))**2).expm() * (c/np.pi)**0.25
            state = operator * spin_coherent(s, np.pi/2,0)
            calc_fisher[j,i] += 4 * variance(jmat(s, 'y'), state) * delta_h
        fisher_eithan[j,i] = N + 1/2*N*(N-1)*(1-np.exp(-c))
        fisher[j,i] = N + (N**2/2) *(1-np.exp(-c))
# chi_args = np.argwhere(chi==[0.1, 0.37])
# for j,N in enumerate(N_arr):
#     plt.plot(chi, fisher[:,i], label='fisher ethan, chi='+str(c))
#     plt.plot(chi, calc_fisher[:,i], label='calc fisher, chi='+str(c))
#     plt.plot(chi, [1/np.sqrt(N + 1/2*N*(N-1)) for N in N_arr], label='final, chi='+str(c))
    plt.plot(chi, fisher_eithan[j,:], label='fisher ethan, N='+str(N))
    plt.plot(chi, fisher[j,:], label='fisher mine, N='+str(N))
    plt.plot(chi, calc_fisher[j,:], label='calc fisher, N='+str(N))
    # plt.plot(chi, [N + 1/2*N*(N-1) for c in chi], label='final fisher')
    # plt.plot(chi, [N for c in chi], label='initial fisher')

    plt.legend()
    plt.show()
#%%
N=6
s=N/2
c=2
sum([(((-c/2) * (num(N+1) - h * qeye(N+1))**2).expm() * (c/np.pi)**0.25)**2  for h in np.linspace(-10*N, 10*N, 20*N+1)])

n=0
print(sum([(exp((-c/2) * (n - h)**2) * (c/np.pi)**0.25)**2 for h in np.linspace(-10*N, 10*N, 2000*N+1)]))
plt.plot(np.linspace(-10*N, 10*N, 20*N+1), [(exp((-c/2) * (n - h)**2) * (c/np.pi)**0.25)**2 for h in np.linspace(-10*N, 10*N, 20*N+1)], '.')
#%%
s=3
N=2*s
for n in range(N+1):
    # print(((N/2) * (1+2*n) - n**2)/2)
    print(np.sqrt((n+1)*(n+2)*(N-n-1)*(N-n))/4)
jmat(s, 'y')**2
#%%

variance(jmat(s, 'y'), (((-c/2) * (-jmat(s, 'z') + (s-1) * qeye(N+1))**2).expm()* spin_coherent(s, np.pi/2,0)).unit())

#%%
for i,c in enumerate(chi):
    plt.plot(N_arr, 1/np.sqrt(fisher[:,i]), label='fisher ethan, sigma='+str(1/np.sqrt(2*c)))
    plt.plot(N_arr, 1/np.sqrt(calc_fisher[:,i]), label='calc fisher, sigma='+str(1/np.sqrt(2*c)))
    plt.plot(N_arr, [1/np.sqrt(N + 1/2*N*(N-1)) for N in N_arr], label='final, sigma='+str(1/np.sqrt(2*c)))
# plt.plot(chi, 1/np.sqrt(fisher), label='fisher ethan')
# plt.plot(chi, 1/np.sqrt(calc_fisher), label='calc fisher')
# plt.plot(chi, [1/np.sqrt(N + 1/2*N*(N-1)) for c in chi], label='final fisher')
# plt.plot(chi, [1/np.sqrt(N) for c in chi], label='initial fisher')

plt.legend()
plt.show()

# %%
operator = ((-chi/2) * (-jmat(s, 'z') + (s-h) * qeye(N+1))**2).expm()
state = operator * spin_coherent(s, np.pi/2,0)
# plot_wigner_sphere(state)
# state = state.unit()
print('initial var='+str(variance(jmat(s, 'y'), spin_coherent(s, np.pi/2,0))))
print('var='+str(variance(jmat(s, 'y'), state)))
calc_fisher = 4 * variance(jmat(s, 'y'), state)
print('calc fisher='+str(calc_fisher))
fisher = N + 1/2*N*(N-1)*(1-np.exp(-chi))
print('fisher='+str(fisher))
plt.plot(sigma, fisher)
plt.plot(sigma, calc_fisher)
# %%
