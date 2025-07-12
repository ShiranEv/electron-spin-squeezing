#%% 
import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
from qutip import *
#%%

def fig_options(ax_nums = False, width = 6, height = 3):
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)

    # Set the box thicker
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)

    # Set ticks inside the box
    # ax.tick_params(direction='in', length=6, width=2, labelsize=16)
    # ax.tick_params(axis='both', which='minor', length=3, width=1, direction='in')
    if not ax_nums:
        # Remove the axes numbers (tick labels)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return fig, ax

def Jz2(N, Nx, sigma):
    xmax = 10 * sigma
    x = np.linspace(-xmax, xmax, Nx)
    dx = x[1] - x[0]
    F0 = np.zeros(len(x))
    for m in range(N + 1):
        F0 += comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigma ** 2)
    px = F0 * 1 / 2 ** N * 1 / np.sqrt(2 * np.pi * sigma ** 2)

    Jz2 = np.trapz(px * Jz2x(N, x, sigma), dx=dx)
    return Jz2

def Jz2x(N, x, sigma):
    F0 = np.zeros(len(x))
    F1 = np.zeros(len(x))
    F2 = np.zeros(len(x))
    for m in range(N + 1):
        F2 += m * m * comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigma ** 2)
        F1 += m * comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigma ** 2)
        F0 += comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigma ** 2)

    px = F0 * 1 / 2 ** N * 1 / np.sqrt(2 * np.pi * sigma ** 2)
    Jz2 = 1 / 2 ** N * 1 / np.sqrt(2 * np.pi * sigma ** 2) * F2 / px - (1 / 2 ** N * 1 / np.sqrt(2 * np.pi * sigma ** 2) * F1 / px) **2
    return Jz2

def Jxx(N, x, sigmaq, sigmac=0):
    sigma = np.sqrt(sigmac ** 2 + sigmaq ** 2)
    F0 = np.zeros(len(x))
    F1 = np.zeros(len(x))
    for m in range(N + 1):
        F1 += comb(N, m) * (N - m) * np.exp(-1 / (8 * sigmaq ** 2) - (x - m - 1 / 2) ** 2 / (2 * sigma ** 2))
        F0 += comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigma ** 2)
    Jx = F1 / F0
    return Jx

def Jx(N, Nx, sigmaq, sigmac=0):
    sigma = np.sqrt(sigmac ** 2 + sigmaq ** 2)
    xmax = 10 * sigma
    x = np.linspace(-xmax, xmax, Nx)
    dx = x[1] - x[0]
    F0 = np.zeros(len(x))
    for m in range(N + 1):
        F0 += comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigma ** 2)
    px = F0 * 1 / 2 ** N * 1 / np.sqrt(2 * np.pi * sigma ** 2)
    Jx = np.trapz(px * Jxx(N, x, sigmaq, sigmac), dx=dx)
    return Jx

def Jy2_mat(n,m,J):
    if n > J or m > J:
        print('n or m is larger than J')
        return 0
    if not ((J*2)-(m*2))%2 == 0 or not ((J*2)-(n*2))%2 == 0:
        print ('J and m or n are not from the same spin class')
        return 0
    elif n == m:
        return 1/2*(J*(J+1)-m**2)
    elif n == m+2:
        return -1/4*np.sqrt((J*(J+1)-(m+2)*(m+1))*(J*(J+1)-m*(m+1)))
    elif n == m-2:
        return -1/4*np.sqrt((J*(J+1)-(m-2)*(m-1))*(J*(J+1)-m*(m-1)))
    else:
        return 0

def QFI_meas_x(N, x, sigmaq):
    # print(x)
    F0 = np.zeros(len(x))
    F1 = np.zeros(len(x))
    for m in range(N + 1):
        F0 += comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigmaq ** 2)
        for n in range(N + 1):
            # print(comb(N, m) * comb(N, n))
            F1 += Jy2_mat(n-N/2,m-N/2,N/2) * np.sqrt(comb(N, m) * comb(N, n)) * np.exp(-(x - m) ** 2 / 4 / sigmaq ** 2)* np.exp(-(x - n) ** 2 / 4 / sigmaq ** 2)
    Jy2x = F1/F0
    return 4*Jy2x

def dphi_meas_qfi(N, Nx, sigmaq):
    x = np.linspace(-10 * sigmaq, N+10 * sigmaq, Nx)
    dphi_x = 1/np.sqrt(QFI_meas_x(N, x, sigmaq))
    dx = x[1] - x[0]
    F0 = np.zeros(len(x))
    for m in range(N + 1):
        F0 += comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigmaq ** 2)
    px = F0 * 1 / 2 ** N * 1 / np.sqrt(2 * np.pi * sigmaq ** 2)
    dphi = np.trapz(px * dphi_x, dx=dx)
    return dphi

def xix(N, x, sigmaq, sigmac=0):
    sigma = np.sqrt(sigmac ** 2 + sigmaq ** 2)
    xi = N * Jz2x(N, x, sigma) / (Jxx(N, x, sigmaq, sigmac) ** 2)
    return xi

def xi(N, Nx, sigmaq, sigmac=0):
    sigma = np.sqrt(sigmac ** 2 + sigmaq ** 2)
    xmax = 10 * sigma
    x = np.linspace(-xmax, N+xmax, Nx)
    dx = x[1] - x[0]
    F0 = np.zeros(len(x))
    for m in range(N + 1):
        F0 += comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigma ** 2)
    px = F0 * 1 / 2 ** N * 1 / np.sqrt(2 * np.pi * sigma ** 2)
    xi = np.trapz(px * xix(N, x, sigmaq, sigmac), dx=dx)

    return xi

def dphi_meas_xi(N, Nx, sigmaq, sigmac=0):
    sigma = np.sqrt(sigmac ** 2 + sigmaq ** 2)
    xmax = 10 * sigma
    x = np.linspace(-xmax, N+xmax, Nx)
    dx = x[1] - x[0]
    F0 = np.zeros(len(x))
    for m in range(N + 1):
        F0 += comb(N, m) * np.exp(-(x - m) ** 2 / 2 / sigma ** 2)
    px = F0 * 1 / 2 ** N * 1 / np.sqrt(2 * np.pi * sigma ** 2)
    dphix = np.sqrt(Jz2x(N, x, sigma) / (Jxx(N, x, sigmaq, sigmac) ** 2))
    dphi = np.trapz(px * dphix, dx=dx)
    return dphi

def meas_based_QFI(N, sigmaq):
    QFI = np.zeros((len(sigmaq), len(N)))
    for i, n in enumerate(N):
       QFI[:,i] = n + 1/2*n*(n-1)*(1-np.exp(-1/(2*sigmaq**2)))
    return QFI

def int_based_QFI(N, chi):
    QFI = np.zeros((len(chi), len(N)))
    for i, n in enumerate(N):
        for j, c in enumerate(chi):
            S_z = jmat(n / 2, 'z')
            S_x = jmat(n / 2, 'x')
            state = (1j * c / 2 * S_z ** 2).expm() * spin_coherent(n/2, np.pi / 2, 0)
            delta = 0.5 * np.arctan(4 * np.sin(c / 2) * np.cos(c / 2) ** (n - 2) / (1 - np.cos(c) ** (n - 2)))
            if np.isnan(delta):
                delta = 0
            Rx=(1j * (np.pi / 2 - delta) * S_x).expm()
            state = Rx * state
            QFI[j, i] = 4* variance(S_z, state)
            # if (n == 20) & (j == 250):
            #     wigner_sphere(state, title="measurement", steps=100, view_1=11, view_2=18)
    return QFI

def measure_based_QFI(N, chi):
    QFI = np.zeros((len(chi), len(N)))
    for i, n in enumerate(N):
        for j, c in enumerate(chi):
            S_z = jmat(n / 2, 'z')
            state = (-c / 2 * S_z ** 2).expm() * spin_coherent(n/2, np.pi / 2, 0)
            state = (1j * (np.pi / 2 ) * jmat(n / 2, 'x')).expm() * state.unit()
            QFI[j, i] = 4* variance(S_z, state)
    return QFI

def int_based_xi(N, chi):
    xi = np.zeros((len(chi), len(N)))
    for i, n in enumerate(N):
        F1 = np.cos(chi/2) ** (-(2*n-2))
        F2 = (n-1)/4*np.cos(chi/2) ** (-(2*n-2))
        F3 = (1-np.cos(chi)**(n-2))**2
        F4 = 16*np.sin(chi/2)**2*np.cos(chi/2)**(2*n-4)
        F5 = 1-np.cos(chi)**(n-2)
        xi[:, i] = F1 - F2 * (np.sqrt(F3 + F4) - F5)
    return xi

#%%
n=100
c_vev = np.linspace(0, 4*3**(1/6)*n**(-2/3), 110)
xi_r = []
for c in c_vev:
    F1 = np.cos(c/2) ** (-(2*n-2))
    F2 = (n-1)/4*np.cos(c/2) ** (-(2*n-2))
    F3 = (1-np.cos(c)**(n-2))**2
    F4 = 16*np.sin(c/2)**2*np.cos(c/2)**(2*n-4)
    F5 = 1-np.cos(c)**(n-2)
    xi_r.append((1 + ((n-1)/4) * (F5 - np.sqrt(F3 + F4)))/(np.cos(c/2)**(2*n-2)))
plt.plot(c_vev, xi_r,'.', label='xi_r')
c_vev[np.argmin(xi_r)]

# %%
n=1000
c=np.pi/2
# fisher
S_z = jmat(n / 2, 'z')
S_x = jmat(n / 2, 'x')
state = (1j * (c / 2) * S_z ** 2).expm() * spin_coherent(n/2, np.pi / 2, 0)
delta = 0.5 * np.arctan(4 * np.sin(c / 2) * np.cos(c / 2) ** (n - 2) / (1 - np.cos(c) ** (n - 2)))
Rx=(1j * (np.pi / 2 - delta) * S_x).expm()
state = Rx * state
print(np.sqrt(1/(4*variance(S_z, state)))) 

#%% plot interaction based dphi vs chi

Nchi = 50
chi_max = 0.5
chi = np.linspace(0, chi_max, Nchi)
N = np.array([20, 200, 2000]).transpose()
dphi_class = 1/np.sqrt(N)

QFI = int_based_QFI(N, chi)
dphi_qfi = np.sqrt(1/QFI)
xi = int_based_xi(N, chi)
dphi_xi = np.sqrt(xi/N)

#%% plot interaction based dphi vs chi
chi = np.linspace(0, 0.5, 5000)
# chi = np.logspace(0, 0.5, 500)

N = np.array([20, 40, 200, 2000])
dphi_class = 1/np.sqrt(N)
dphi_heisenberg = 1/N
dphi_qfi = np.zeros((len(N), len(chi)))
dphi_xi = np.zeros((len(N), len(chi)))
for i, n in enumerate(N):
    s = n/2
    for j, c in enumerate(chi):
        F1 = np.cos(c/2) ** (-(2*n-2))
        F2 = (n-1)/4*np.cos(c/2) ** (-(2*n-2))
        F3 = (1-np.cos(c)**(n-2))**2
        F4 = 16*np.sin(c/2)**2*np.cos(c/2)**(2*n-4)
        F5 = 1-np.cos(c)**(n-2)
        dphi_xi[i, j] = np.sqrt((F1 - F2 * (np.sqrt(F3 + F4) - F5))/n)
        dphi_qfi[i, j] = 1/np.sqrt(n + (n*(n-1)/4) * (np.sqrt(F3 + F4) + F5))


fig, ax = fig_options(ax_nums = True)
# plt.ylim([-0.02, 0.25])
plt.ylim([0.0004, 0.45])
plt.xlim(chi[4], chi[-1])
colors = ['black', 'tab:blue', 'tab:red', 'tab:orange']

for i, n in enumerate(N):
    plt.hlines(dphi_class[i], 0, 0.5, color=colors[i], linestyles=':')
    plt.hlines(dphi_heisenberg[i], 0, 0.5, color=colors[i], linestyles='-.')
    plt.plot(chi, dphi_qfi[i, :], color=colors[i], linestyle = "--")
    plt.plot(chi, dphi_xi[i, :], color=colors[i])
    plt.plot(chi[np.argmin(dphi_xi[i, :])], np.min(dphi_xi[i, :]), 'o', color=colors[i])
    print('N = {:.0f}, chi_opt = {:.2f}, delta_phi = {:.3f}'.format(n, chi[np.argmin(dphi_xi[i, :])], np.min(dphi_xi[i, :])))
# ax.set_yticks([0, 0.1, 0.2, 0.3])
# ax.set_xscale('log')
ax.set_yscale('log')
optimal_chi = [chi[np.argmin(dphi_xi[i, :])] for i in range(len(N))]

# ax.set_xticks(optimal_chi)
# ax.set_xticklabels([f"{chi:.3f}" for chi in optimal_chi])
plt.savefig('figs/interaction_dphi_chi_log.svg')
plt.show()

# %% 
for i, n in enumerate(N):
    print(dphi_xi[i, 0], dphi_xi[i, np.abs(chi - 0.122).argmin()], dphi_xi[i, np.abs(chi - optimal_chi[i]).argmin()])


# %% plot interaction based dphi vs N
N_log = np.logspace(1, 3.4, 100, dtype=int)
# chi = np.array([0.02, 0.07, 0.26])
dphi_class = 1/np.sqrt(N_log)
dphi_heisenberg = 1/N_log
dphi_qfi = np.zeros((len(N_log), len(optimal_chi)))
dphi_xi = np.zeros((len(N_log), len(optimal_chi)))
for i, n in enumerate(N_log):
    s = n/2
    for j, c in enumerate(optimal_chi):
        F1 = np.cos(c/2) ** (-(2*n-2))
        F2 = (n-1)/4*np.cos(c/2) ** (-(2*n-2))
        F3 = (1-np.cos(c)**(n-2))**2
        F4 = 16*np.sin(c/2)**2*np.cos(c/2)**(2*n-4)
        F5 = 1-np.cos(c)**(n-2)
        dphi_xi[i, j] = np.sqrt((F1 - F2 * (np.sqrt(F3 + F4) - F5))/n)
        dphi_qfi[i, j] = 1/np.sqrt(n + (n*(n-1)/4) * (np.sqrt(F3 + F4) + F5))

fig, ax = fig_options(ax_nums= True)
plt.ylim([0.0005, 0.45])
plt.xlim(N_log[0], N_log[-1])
# colors = ['purple', 'tab:orange', 'green']
for j, c in enumerate(optimal_chi):
    plt.plot(N_log, dphi_qfi[:, j], color=colors[j], linestyle = "--")
    plt.plot(N_log, dphi_xi[:, j], color=colors[j],)
    plt.plot(N_log, dphi_class, color='gray', linestyle=':')
    plt.plot(N_log, dphi_heisenberg, color='gray', linestyle='-.')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(N)
ax.set_xticklabels(N)
plt.savefig('figs/int_dphi_N.svg')
plt.show()

# %% interaction based optimal chi for ramsy xi squeezing parameter vs chi
chi_opt_int = np.linspace(0, 0.5, 5000)
if 20 not in N_log:
    print('N=20 not in N')
    N_log = np.sort(np.append(N_log, 20))
if 200 not in N_log:
    print('N=200 not in N')
    N_log = np.sort(np.append(N_log, 200))
if 2000 not in N_log:
    print('N=2000 not in N')
    N_log = np.sort(np.append(N_log, 2000))
dphi_xi_opt_int = np.zeros((len(N_log), len(chi_opt_int)))
for i, n in enumerate(N_log):
    # print('n='+str(n))
    s = n/2
    for j, c in enumerate(chi_opt_int):
        F1 = np.cos(c/2) ** (-(2*n-2))
        F2 = (n-1)/4*np.cos(c/2) ** (-(2*n-2))
        F3 = (1-np.cos(c)**(n-2))**2
        F4 = 16*np.sin(c/2)**2*np.cos(c/2)**(2*n-4)
        F5 = 1-np.cos(c)**(n-2)
        dphi_xi_opt_int[i, j] = np.sqrt((F1 - F2 * (np.sqrt(F3 + F4) - F5))/n)

fig, ax = fig_options(ax_nums = True, height=2)
plt.xlim(N_log[0], N_log[-1])
plt.ylim([0.008, 0.5])
plt.plot(N_log, chi_opt_int[np.argmin(dphi_xi_opt_int, 1)],color='gray', linewidth=2.0)
# plt.plot(N, 2/((N)**(2/3)), linewidth=2.0)
# colors = ['black', 'tab:blue', 'tab:red']
for i, n in enumerate(N):
    plt.plot(n, optimal_chi[i], 'o', color=colors[i],  linewidth=2.0)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_yticks(optimal_chi)
# ax.set_yticklabels(np.round(optimal_chi, 3))
ax.set_xticks(N)
ax.set_xticklabels(N)
plt.savefig('figs/int_chiopt_N.svg')
plt.show()

#%% plot measurement based dphi vs chi
chi = np.linspace(0.0001, 10, 1000)
sigma = 1/(np.sqrt(2*chi))
# N = np.array([20, 100, 1000]).transpose()
N = np.array([40]).transpose()
dphi_class = 1/np.sqrt(N)
dphi_xi = np.zeros([len(N), len(sigma)])
dphi_qfi = np.zeros([len(N), len(sigma)])
for i, n in enumerate(N):
    print('n='+str(n))
    n = int(n)
    Nx = max(4*n+1, 1001)
    # dx = 1/comb(int(n), int(n/2))
    x = np.linspace(0, n, Nx)
    # if dx > 0:
    for j, s in enumerate(sigma):
        # dx = 1/comb(int(n), int(n/2))
        x = np.linspace(0, n, Nx)
        c = 1/(2*s**2)
        avg_x = np.zeros(len(x))
        F0 = np.zeros(len(x))
        F1 = np.zeros(len(x))
        F2 = np.zeros(len(x))
        for m in range(n + 1):
            # print(m)
            if n<400:
                comb_val = comb(n, m)/comb(int(n), int(n/2))
            else:
                comb_val = np.exp(-(2/n)*(m-n/2)**2)*np.sqrt(2/(np.pi*n))
            
            a =  comb_val * np.exp(-c*(x - m) ** 2)
            F2 += a * m**2 
            F1 += a * m 
            avg_x += comb_val * (n - m) * np.exp(-1 / (8 * s ** 2) - c* (x - m - 1 / 2)** 2)
            F0 += a
        var_z = F0 * F2 - F1**2
        dphi_xi[i, j] = np.sum(F0 * np.sqrt(var_z)/np.abs(avg_x))/np.sum(F0)
        dphi_qfi[i, j] = 1/np.sqrt(n + (n**2/2) * (1-np.exp(-c)))

fig, ax = fig_options(ax_nums = True)
# plt.ylim([-0.02, 0.25])
# plt.ylim([0.0005, 0.45])
# plt.xlim(chi[0], chi[-1])
colors = ['black', 'tab:blue', 'tab:red']

for i, n in enumerate(N):
    plt.hlines(dphi_class[i], 0, chi[-1], color=colors[i], linestyles=':')
    # plt.hlines(dphi_heisenberg[i], 0, chi_max, color=colors[i], linestyles='-.')
    plt.plot(chi, dphi_qfi[i, :], color=colors[i], linestyle = "--")
    plt.plot(chi, dphi_xi[i, :], color=colors[i])
    plt.plot(chi[np.argmin(dphi_xi[i, :])], np.min(dphi_xi[i, :]), 'o', color=colors[i])
    print('N = {:.0f}, chi_opt = {:.2f}, delta_phi = {:.3f}'.format(n, chi[np.argmin(dphi_xi[i, :])], np.min(dphi_xi[i, :])))
# ax.set_yticks([0, 0.1, 0.2, 0.3])
# ax.set_xscale('log')
# ax.set_yscale('log')
optimal_chi = np.array([chi[np.argmin(dphi_xi[i, :])] for i in range(len(N))])

# ax.set_xticks(optimal_chi)
# ax.set_xticklabels([f"{chi:.3f}" for chi in optimal_chi])
# plt.savefig('figs/measure_dphi_chi.svg')
plt.show()

# %% plot measurement based dphi vs N
# N_log = np.logspace(1, np.log10(1485), 50, dtype=int)
N_log = np.arange(15, 30)
sigma = np.array([3, 1])
sigma = np.append(sigma, 1/np.sqrt(2*optimal_chi))
dphi_class = 1/np.sqrt(N_log)
dphi_heisenberg = 1/N_log
dphi_qfi = np.zeros((len(N_log), len(sigma)))
dphi_xi = np.zeros((len(N_log), len(sigma)))
for i, n in enumerate(N_log):
    print('n='+str(n))
    n = int(n)
    Nx = max(50*n+1, 1001)
    # dx = 1/comb(int(n), int(n/2))
    x = np.linspace(0, n, Nx)
    # if dx > 0:
    for j, s in enumerate(sigma):
        c = 1/(2*s**2)
        avg_x = np.zeros(len(x))
        F0 = np.zeros(len(x))
        F1 = np.zeros(len(x))
        F2 = np.zeros(len(x))
        for m in range(n + 1):
            # print(m)
            if n<400:
                comb_val = comb(n, m)/comb(int(n), int(n/2))
            else:
                comb_val = np.exp(-(2/n)*(m-n/2)**2)*np.sqrt(2/(np.pi*n))
            
            a =  comb_val * np.exp(-c*(x - m) ** 2)
            F2 += a * m**2 
            F1 += a * m 
            avg_x += comb_val * (n - m) * np.exp(-1 / (8 * s ** 2) - c* (x - m - 1 / 2)** 2)
            F0 += a
        var_z = F0 * F2 - F1**2
        print(avg_x)
        dphi_xi[i, j] = np.sum(F0 * np.sqrt(var_z)/avg_x)/np.sum(F0)
        dphi_qfi[i, j] = 1/np.sqrt(n + (n**2/2) * (1-np.exp(-c)))
        

fig, ax = fig_options(width=3, height=2)
plt.ylim([0.05, 0.25])
plt.xlim(N_log[0], N_log[-1])
colors = ['tab:orange', 'green', 'black', 'tab:blue', 'black']
for j, s in enumerate(sigma):
    plt.plot(N_log, dphi_qfi[:, j], color=colors[j], linestyle = "--", linewidth=2.0)
    plt.plot(N_log, dphi_xi[:, j], color=colors[j], linewidth=2.0)
    plt.plot(N_log, dphi_class, color='gray', linestyle=':', linewidth=2.0)
    plt.plot(N_log, dphi_heisenberg, color='gray', linestyle='-.', linewidth=2.0)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xticks(N)
# ax.set_xticklabels(N)
plt.savefig('figs/measure_dphi_N_zoom.svg')
plt.show()
        
# %% measure based optimal chi for ramsy xi squeezing parameter vs chi
chi_opt = np.linspace(1.897, 2.115, 500)
sigma = 1/(np.sqrt(2*chi_opt))
for i in range(len(N)):
    if N[i] not in N_log:
        print(N[i])
        N_log = np.sort(np.append(N_log, N[i]))
dphi_xi_opt = np.zeros((len(N_log), len(chi_opt)))
for i, n in enumerate(N_log):
    print('i='+str(i)+', n='+str(n))
    n = int(n)
    Nx = max(4*n+1, 1001)
    # dx = 1/comb(int(n), int(n/2))
    x = np.linspace(0, n, Nx)
    # if dx > 0:
    for j, s in enumerate(sigma):
        c = 1/(2*s**2)
        avg_x = np.zeros(len(x))
        F0 = np.zeros(len(x))
        F1 = np.zeros(len(x))
        F2 = np.zeros(len(x))
        for m in range(n + 1):
            # print(m)
            if n<400:
                comb_val = comb(n, m)/comb(int(n), int(n/2))
            else:
                comb_val = np.exp(-(2/n)*(m-n/2)**2)*np.sqrt(2/(np.pi*n))
            
            a =  comb_val * np.exp(-c*(x - m) ** 2)
            F2 += a * m**2 
            F1 += a * m 
            avg_x += comb_val * (n - m) * np.exp(-1 / (8 * s ** 2) - c* (x - m - 1 / 2)** 2)
            F0 += a
        var_z = F0 * F2 - F1**2

        dphi_xi_opt[i, j] = np.sum(F0 * np.sqrt(var_z)/avg_x)/np.sum(F0)

fig, ax = fig_options(height=2)
# plt.xlim(N_log[0], N_log[-1])
# plt.ylim(1.9, chi_opt[-1])
colors = ['black', 'tab:blue', 'tab:red']

plt.plot(N_log, chi_opt[np.argmin(dphi_xi_opt, 1)],linewidth=2.0, color='gray')
# plt.plot(N_log, 2.1-2/N_log*(3/2), linewidth=2.0)
for i, n in enumerate(N):
    plt.plot(n, chi_opt[np.argmin(dphi_xi_opt, 1)][np.argwhere(N_log==n)[0,0]], 'o', color=colors[i],  linewidth=2.0)
ax.set_xscale('log')
# ax.set_yscale('log')
optimal_chi = [chi_opt[np.argmin(dphi_xi_opt, 1)][np.argwhere(N_log==n)[0,0]] for n in N]
ax.set_yticks([1.9, 2, 2.1])
ax.set_yticklabels([1.9, 2, 2.1])
ax.set_xticks(N)
ax.set_xticklabels(N)
plt.savefig('figs/measure_chiopt_N.svg')
plt.show()





#%%
#send the function and everything they need before
# N = np.arange(10, 2020, 200)
N = np.logspace(1, 2.4, 20, dtype=int)
N = np.sort(np.append(N, 20))
sigma = np.array([40, 3, 1])
chi = 1/(2*sigma**2)
dphi_qfi = np.zeros((len(N), len(chi)))
dphi_xi = np.zeros((len(N), len(chi)))
for i, n in enumerate(N):
    print('n='+str(n))
    s = n/2
    for j, c in enumerate(chi):
        h_vec = np.linspace(0, n+1, int(np.ceil(2*n)))
        p_h = np.array([sum([comb(n, m) * np.exp(-c*(m-h)**2) for m in range(n+1)]) for h in h_vec])
        p_h = p_h/sum(p_h)
        state_h = [(((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm() * spin_coherent(s, np.pi/2,0)).unit() for h in h_vec]
        dphi_xi[i, j] = sum(p_h * [np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state) for state in state_h])
        dphi_qfi[i, j] = sum(p_h * [np.sqrt(1/(4*variance(jmat(s, 'y'), state))) for state in state_h])


# %% 
fig, ax = fig_options(ax_nums= True, height=2)
plt.ylim([0.005, 0.45])
plt.xlim(N[0]-1, N[-1])
colors = ['purple', 'tab:orange', 'green']
for i, c in enumerate(chi):
    # plt.plot(N, 2/N, color='tab:blue', linestyle=':', linewidth=2.0, label='SQL, chi={}'.format(c))
    plt.plot(N, dphi_qfi[:, i], color=colors[i], label='QFI, chi={}'.format(c))
    plt.plot(N, dphi_xi[:, i], color=colors[i], marker='+', linestyle='None', label='squeez_par, chi={}'.format(c))
# plt.legend()
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_yticks([0, 0.1, 0.2, 0.3])
# ax.set_xticks([10, 20, 30, 100,1000,2000])
# ax.set_xticklabels([])
plt.savefig('figs/measure_dphi_N.svg')
plt.show()

# %% plot interaction based optimal chi for ramsy xi squeezing parameter vs chi
chi_opt = np.linspace(1.8, 2.2, 50)
N = np.logspace(1, 1.8, 10, dtype=int)
N = np.sort(np.append(N, 20))
N = np.sort(np.append(N, 30))
dphi_xi_opt = np.zeros((len(N), len(chi_opt)))
for i, n in enumerate(N):
    print('n='+str(n))
    s = n/2
    h_vec = np.linspace(0, n+1, int(np.ceil(4*n)))
    for j, c in enumerate(chi_opt):
        p_h = np.array([sum([comb(n, m) * np.exp(-c*(m-h)**2) for m in range(n+1)]) for h in h_vec])
        p_h = p_h/sum(p_h)
        state_h = [(((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm() * spin_coherent(s, np.pi/2,0)).unit() for h in h_vec]
        dphi_xi_opt[i, j] = sum(p_h * [np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state) for state in state_h])

# %%
fig, ax = fig_options(ax_nums = True, height=2)
plt.xlim(N[0]-1, N[-1])
plt.plot(N, chi_opt[np.argmin(dphi_xi_opt, 1)], linewidth=2.0)
# plt.plot(N, 2/((N)**(2/3)), linewidth=2.0)
colors = ['black', 'tab:blue', 'tab:red']
for i, n in enumerate([10, 20, 30]):
 plt.plot(n, chi_opt[np.argmin(dphi_xi_opt, 1)][np.argwhere(N==n)[0,0]], 'o', color=colors[i],  linewidth=2.0)
# ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_yticks([0.1, 0.26, 0.4])
# ax.set_yticklabels([0.1, 0.26, 0.4])
plt.savefig('figs/measure_chiopt_N.svg')
plt.show()
#%% plot measur based dphi vs sigma
Nchi = 50
chi_max = 2.4
chi = np.linspace(0.004, chi_max, Nchi)
sigma = 1/(np.sqrt(2*chi))

N = np.array([10, 20, 30]).transpose()
dphi_class = 1/np.sqrt(N)

dphi_xi = np.zeros([len(N), len(sigma)])
dphi_qfi = np.zeros([len(N), len(sigma)])

for i, n in enumerate(N):
    print('n='+str(n))
    s = n/2
    for j, c in enumerate(chi):
        h_vec = np.linspace(0, n+1, int(np.ceil(2*c*n)))
        p_h = np.array([sum([comb(n, m) * np.exp(-c*(m-h)**2) for m in range(n+1)]) for h in h_vec])
        p_h = p_h/sum(p_h)
        state_h = [(((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm() * spin_coherent(s, np.pi/2,0)).unit() for h in h_vec]
        dphi_xi[i, j] = sum(p_h * [np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state) for state in state_h])
        dphi_qfi[i, j] = sum(p_h * [np.sqrt(1/(4*variance(jmat(s, 'y'), state))) for state in state_h])
        # state = (((-c/2) * jmat(s, 'z')**2).expm() * spin_coherent(s, np.pi/2,0)).unit()
        # dphi_xi[i, j] = np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state)
        # dphi_qfi[i, j] = np.sqrt(1/(4*variance(jmat(s, 'y'), state)))

#%% plot measur based dphi vs sigma
Nchi = 50
chi_max = 2.4
chi = np.linspace(0.004, chi_max, Nchi)
sigma = 1/(np.sqrt(2*chi))

N = np.array([10, 20, 30]).transpose()
dphi_class = 1/np.sqrt(N)

dphi_xi = np.zeros([len(N), len(sigma)])
dphi_qfi = np.zeros([len(N), len(sigma)])

for i, n in enumerate(N):
    print('n='+str(n))
    s = n/2
    for j, s in enumerate(sigma):
        Nx = 1001
        xmax = 10 * s
        x = np.linspace(-xmax, n+xmax, Nx)
        dx = x[1] - x[0]
        F0 = np.zeros(len(x))
        F1 = np.zeros(len(x))
        F1_ = np.zeros(len(x))
        F2 = np.zeros(len(x))
        for m in range(n + 1):
            F2 += m * m * comb(n, m) * np.exp(-(x - m) ** 2 / 2 / s ** 2)
            F1 += m * comb(n, m) * np.exp(-(x - m) ** 2 / 2 / s ** 2)
            F1_ += comb(n, m) * (n - m) * np.exp(-1 / (8 * s ** 2) - (x - m - 1 / 2) ** 2 / (2 * s ** 2))
            F0 += comb(n, m) * np.exp(-(x - m) ** 2 / 2 / s ** 2)
        px = F0 * 1 / 2 ** n * 1 / np.sqrt(2 * np.pi * s ** 2)
        Jz2 = 1 / 2 ** n * 1 / np.sqrt(2 * np.pi * s ** 2) * F2 / px - (1 / 2 ** n * 1 / np.sqrt(2 * np.pi * s ** 2) * F1 / px) **2
        Jx = F1_ / F0
        dphix = np.sqrt(Jz2 / (Jx ** 2))
        dphi_xi[i, j] = np.trapz(px * dphix, dx=dx)
#%%
fig, ax = fig_options(ax_nums = True)
plt.ylim([-0.02, 0.35])
plt.xlim(chi[0], chi[-1])
colors = ['black', 'tab:blue', 'tab:red']

for i, n in enumerate(N):
    plt.hlines(dphi_class[i], 0, chi_max, color=colors[i], linestyles=':', linewidth=2.0, label='SQL, N={}'.format(n))
    plt.plot(chi, dphi_qfi[i, :], color=colors[i], linestyle = "--", linewidth=2.0, label='QFI, N={}'.format(n))
    plt.plot(chi, dphi_xi[i, :], color=colors[i],  linewidth=2.0, label='squeez_par, N={}'.format(n))
    plt.plot(chi[np.argmin(dphi_xi[i, :])], np.min(dphi_xi[i, :]), 'o', color=colors[i],  linewidth=2.0)

# plt.legend()
ax.set_yticks([0, 0.1, 0.2, 0.3])
# ax.set_xticks([0.1, 0.3, 0.5, 0.7])
# plt.savefig('figs/measure_dphi_chi.svg')
plt.show()
#%% 
fig, ax = fig_options(ax_nums = True)
# plt.ylim([0, 0.5])
plt.xlim(0, np.max(sigma))
colors = ['black', 'tab:blue', 'tab:red']

for i, c in enumerate(N):
    plt.hlines(dphi_class[i], 0, np.max(sigma), color=colors[i], linestyles=':', linewidth=2.0, label='SQL, N={}'.format(c))
    plt.plot(sigma, dphi_qfi[i, :], color=colors[i], linestyle = "--", linewidth=2.0, label='QFI, N={}'.format(c))
    plt.plot(sigma, dphi_xi[i, :], color=colors[i],  linewidth=2.0, label='squeez_par, N={}'.format(c))
# plt.legend()
# ax.set_yticks([0, 0.25, 0.5])
# ax.set_xticks([0.1, 0.3, 0.5, 0.7])
plt.savefig('figs/measure_dphi_sigma.svg')
plt.show()

# %%
s=10
n=2*s
c=0.3
sigma = 1/(np.sqrt(2*c))
h_vec = np.arange(0, n+1, 1)
dphi = 0
p_h = np.abs(spin_coherent(s, np.pi/2, 0)[:,0])**2

state = (((-c/2) * jmat(s, 'z')**2).expm() * spin_coherent(s, np.pi/2, 0)).unit()
phase_f = 1/np.sqrt(4 * variance(jmat(s, 'y'), state))
print('QFI phase = {}'.format(phase_f))
sq_phase = np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state)
print('squeezing phase = {}'.format(sq_phase))
# dphi_xi = dphi_meas_xi(n, 10000, sigma, sigma)
# print('xi phase = {}'.format(dphi_xi))
P=0
for h in h_vec:
    operator = ((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm()
    state = (((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm() * spin_coherent(s, np.pi/2,0)).unit()
    # print('h='+str(h)+' p_h = {}'.format(p_h[h]))
    # P += p_h[h]
    dphi += p_h[h] * np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state)
state_h = [(((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm() * spin_coherent(s, np.pi/2,0)).unit() for h in h_vec]
dphi_arr = sum(p_h * [np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state) for state in state_h])
print('dphi = {}'.format(dphi_arr))   
# print('P = {}'.format(P))
print('xi phase avg = {}'.format(dphi))







# %% plot measure based dphi vs N
sigma = np.array([40, 3, 1])
chi = 1/(2*sigma**2)
dphi_qfi = np.zeros((len(N), len(chi)))
dphi_xi = np.zeros((len(N), len(chi)))
for i, n in enumerate(N):
    print('n='+str(n))
    s = n/2
    h_vec = np.arange(0, n+1, 1)
    p_h = np.abs(spin_coherent(s, np.pi/2, 0)[:,0])**2
    for j, c in enumerate(chi):
        state_h = [(((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm() * spin_coherent(s, np.pi/2,0)).unit() for h in h_vec]
        dphi_xi[i, j] = sum(p_h * [np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state) for state in state_h])
        dphi_qfi[i, j] = sum(p_h * [np.sqrt(1/(4*variance(jmat(s, 'y'), state))) for state in state_h])

fig, ax = fig_options(ax_nums= True)
plt.ylim([0.0005, 0.45])
plt.xlim(N[0]-1, N[-1])
colors = ['purple', 'tab:orange', 'green']
for i, c in enumerate(chi):
    plt.plot(N, dphi_qfi[:, i], color=colors[i], label='QFI, chi={}'.format(c))
    plt.plot(N, dphi_xi[:, i], color=colors[i], marker='+', linestyle='None', label='squeez_par, chi={}'.format(c))
ax.set_xscale('log')
ax.set_yscale('log')
# plt.savefig('figs/measure_dphi_N.svg')
plt.show()

# %% 
# chi = np.array([0.05, 0.5, 3]).transpose()
# sigma = 1/(np.sqrt(2*chi))
# sigma = np.array([40, 1, 0.4]).transpose()

Nx = 1001

# dphi_xi = np.zeros((len(sigma), len(N)))
R = 1e-12

for j,s in enumerate(sigma):
        for i, n in enumerate(N):
            dphi_xi[j,i] = dphi_meas_xi(n, Nx, sigmaq[j], sigmac[j])


QFI = measure_based_QFI(N, 1/(2*sigma**2))
dphi_qfi = np.sqrt(1/QFI)  


# %% 
fig, ax = fig_options(ax_nums = True)
plt.ylim([-0.02, 0.35])
plt.xlim(N[0], 61)
colors = ['purple', 'tab:orange', 'green']
for i, c in enumerate(sigma):
    # plt.plot(N_xi, 1/np.sqrt(N_xi), color='tab:blue', linestyle=':', linewidth=2.0, label='SQL, chi={}'.format(c))
    plt.plot(N, dphi_qfi[i,:], color=colors[i], linestyle = "--", linewidth=2.0, label='QFI, chi={}'.format(c))
    plt.plot(N, dphi_xi[i,:], color=colors[i],  linewidth=2.0, label='squeez_par, chi={}'.format(c))

# plt.legend()
ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.set_xticks([10, 20, 30, 40,50,60])
plt.savefig('figs/measure_dphi_N.svg')
plt.show()




# %%
Nchi = 20

chi = np.linspace(2, 10, Nchi)
sigma = 1/(np.sqrt(2*chi))

N = np.arange(8, 62, 10)

dphi_xi_opt = np.zeros((len(N), len(sigma)))
for i, n in enumerate(N):
    print('n='+str(n))
    s = n/2
    h_vec = np.arange(0, n+1, 1)
    p_h = np.abs(spin_coherent(s, np.pi/2, 0)[:,0])**2
    for j,c in enumerate(chi):
        state_h = [(((-c/2) * (s * qeye(n+1) - jmat(s, 'z') - h*qeye(n+1))**2).expm() * spin_coherent(s, np.pi/2,0)).unit() for h in h_vec]
        dphi_xi_opt[i, j] = sum(p_h * [np.sqrt(variance(jmat(s, 'z'), state))/expect(jmat(s, 'x'), state) for state in state_h])
      
# %%
fig, ax = fig_options(ax_nums = True, height=2)
plt.xlim(N[0], N[-1])
plt.plot(N, chi[np.argmin(dphi_xi_opt, 1)],'.', linewidth=2.0)
colors = ['black', 'tab:blue', 'tab:red']
N_3 = np.array([10, 20, 30])
for i, n in enumerate(N_3):
    plt.plot(n, chi[np.argmin(dphi_xi_opt, 0)][np.argwhere(N==n)[0,0]], 'o', color=colors[i],  linewidth=2.0)


ax.set_xticks([10, 20, 30, 40,50,60])
plt.savefig('figs/measure_chiopt_N.svg')
plt.show()
# %%
fig, ax = fig_options(ax_nums = True, height=2)
plt.xlim(N[0], N[-1])
plt.plot(N, 1/sigma[np.argmin(dphi_xi_opt, 0)],'.', linewidth=2.0)
colors = ['black', 'tab:blue', 'tab:red']
N_3 = np.array([10, 20, 30])
for i, n in enumerate(N_3):
    plt.plot(n, 1/sigma[np.argmin(dphi_xi_opt, 0)][np.argwhere(N==n)[0,0]], 'o', color=colors[i],  linewidth=2.0)


ax.set_xticks([10, 20, 30, 40,50,60])
# plt.savefig('figs/interaction_chiopt_N.svg')
plt.show()
# %%
