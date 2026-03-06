import numpy as np
import matplotlib.pyplot as plt

np.random.seed(592)

n    = 1000       # time horizon
K    = 2          # number of arms
gap  = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
c_eg = 50         # epsilon-greedy constant

#
# ALGORITHM 1: EXPLORE-THEN-COMMIT
#

def explore_then_commit(Delta, n, simulations=100):
    m = int(np.ceil(n ** (2/3)))  # exploration rounds *per arm*
    means = [0.0, -Delta]         # arm 0 is optimal
    regrets = np.zeros(simulations)

    for sim in range(simulations):
        N_i = [0, 0]
        sum_rewards  = [0.0, 0.0]
        total_regret = 0.0

        # --- Exploration phase: alternate arms for m pulls each ---
        for t in range(2 * m):
            arm = t % 2
            reward = np.random.normal(means[arm], 1)
            sum_rewards[arm] += reward
            N_i[arm]         += 1
            # Regret this round = gap between optimal and chosen arm
            total_regret += means[0] - means[arm]

        # Commit to the arm with the highest *empirical* mean
        emp_means = [sum_rewards[i] / N_i[i] for i in range(K)]
        best_arm  = int(np.argmax(emp_means))

        # --- Exploitation phase: play best arm for remaining rounds ---
        for t in range(2 * m, n):
            # No randomness in arm choice anymore — always best_arm
            total_regret += means[0] - means[best_arm]

        regrets[sim] = total_regret

    avg_regret = np.mean(regrets)
    std_error  = np.std(regrets) / np.sqrt(simulations)
    return avg_regret, std_error


# ALGORITHM 2: SUCCESSIVE ELIMINATION
#
# Bonus: b_i(t) = sqrt(2 log(n) / N_i(t))
# UCB_i = empirical_mean_i + b_i
# LCB_i = empirical_mean_i - b_i


def successive_elimination(Delta, n, simulations=100):
    means   = [0.0, -Delta]
    regrets = np.zeros(simulations)

    for sim in range(simulations):
        active = list(range(K))  # all arms start active
        N_i = [0, 0]
        sum_rewards = [0.0, 0.0]
        total_regret = 0.0
        t = 0

        while t < n:
            # Play every active arm exactly once this round
            for arm in list(active):
                if t >= n:
                    break
                reward = np.random.normal(means[arm], 1)
                sum_rewards[arm] += reward
                N_i[arm] += 1
                total_regret += means[0] - means[arm]
                t+= 1

            # Compute confidence intervals for all active arms
            emp = {arm: sum_rewards[arm] / N_i[arm] for arm in active}
            b   = {arm: np.sqrt(2 * np.log(n) / N_i[arm]) for arm in active}
            ucb = {arm: emp[arm] + b[arm] for arm in active}
            lcb = {arm: emp[arm] - b[arm] for arm in active}

            # Eliminate arm i if some arm j has LCB_j > UCB_i
            # (i.e., we're confident j is strictly better than i)
            to_remove = [
                arm_i for arm_i in active
                if any(lcb[arm_j] > ucb[arm_i] for arm_j in active)
            ]
            for arm in to_remove:
                active.remove(arm)

            # Once only one arm remains, commit to it for the rest
            if len(active) == 1:
                remaining = active[0]
                while t < n:
                    total_regret += means[0] - means[remaining]
                    t            += 1
                break

        regrets[sim] = total_regret

    avg_regret = np.mean(regrets)
    std_error  = np.std(regrets) / np.sqrt(simulations)
    return avg_regret, std_error


#
# ALGORITHM 3: EPSILON-GREEDY
#

def epsilon_greedy(Delta, n, simulations=100, c=50):
    means   = [0.0, -Delta]
    regrets = np.zeros(simulations)

    for sim in range(simulations):
        N_i          = [0, 0]
        sum_rewards  = [0.0, 0.0]
        total_regret = 0.0

        for t in range(1, n + 1):
            eps_t = min(1.0, c / t)

            if np.random.random() < eps_t:
                # Explore uniformly at random
                arm = np.random.randint(0, K)
            else:
                # Exploit: handle edge case where an arm has 0 pulls
                if N_i[0] == 0:
                    arm = 0
                elif N_i[1] == 0:
                    arm = 1
                else:
                    emp_means = [sum_rewards[i] / N_i[i] for i in range(K)]
                    arm = int(np.argmax(emp_means))

            reward = np.random.normal(means[arm], 1)
            sum_rewards[arm] += reward
            N_i[arm] += 1
            total_regret += means[0] - means[arm]

        regrets[sim] = total_regret

    avg_regret = np.mean(regrets)
    std_error  = np.std(regrets) / np.sqrt(simulations)
    return avg_regret, std_error


# MONTE CARLO SIMULATIONS
# Run 100 simulations per algorithm per Delta value

print("Running Monte Carlo simulations (100 runs per setting)...")

etc_avg, etc_err = [], []
se_avg,  se_err  = [], []
eg_avg,  eg_err  = [], []

for delta in gap:
    print(f"  Δ = {delta:.2f}", end="  |  ")

    r, e = explore_then_commit(delta, n, simulations=100)
    etc_avg.append(r); etc_err.append(e)
    print(f"ETC={r:.1f}", end="  ")

    r, e = successive_elimination(delta, n, simulations=100)
    se_avg.append(r); se_err.append(e)
    print(f"SE={r:.1f}", end="  ")

    r, e = epsilon_greedy(delta, n, simulations=100, c=c_eg)
    eg_avg.append(r); eg_err.append(e)
    print(f"EG={r:.1f}")


# THEORETICAL BOUNDS
# ETC:  R(n) <= Delta*m + Delta*(n-m)*exp(-m*Delta^2/4)
# SE:   R(n) <= sqrt(K*n*log(n))   [constant in Delta]
# EG:   R(n) <= c*Delta + Delta*n/c

def etc_bound(Delta, n):
    m = int(np.ceil(n ** (2/3)))
    return Delta * m + Delta * (n - m) * np.exp(-m * Delta**2 / 4)

def se_bound(n, K=2):
    return np.sqrt(K * n * np.log(n))

def eg_bound(Delta, n, c=50):
    return c * Delta + Delta * n / c

theo_etc = [etc_bound(d, n)   for d in gap]
theo_se  = [se_bound(n, K=2)] * len(gap)   # only flat line????
theo_eg  = [eg_bound(d, n, c_eg) for d in gap]


# PLOTTING
# Two side-by-side panels:
#   Left:  empirical regret with error bars
#   Right: empirical + theoretical bounds overlaid

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Shared style
colors = {'etc': 'steelblue', 'se': 'darkorange', 'eg': 'forestgreen'}

# Panel 1: Empirical only 
ax = axes[0]
ax.errorbar(gap, etc_avg, yerr=etc_err, label='ETC',
            color=colors['etc'], marker='o', capsize=3, linewidth=1.5)
ax.errorbar(gap, se_avg,  yerr=se_err,  label='Successive Elim.',
            color=colors['se'],  marker='s', capsize=3, linewidth=1.5)
ax.errorbar(gap, eg_avg,  yerr=eg_err,  label='ε-Greedy',
            color=colors['eg'],  marker='^', capsize=3, linewidth=1.5)
ax.set_xlabel('Gap Δ', fontsize=12)
ax.set_ylabel('Expected Regret', fontsize=12)
ax.set_title('Empirical Regret vs Gap (n=1000)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Empirical + theoretical 
ax = axes[1]
ax.errorbar(gap, etc_avg, yerr=etc_err, label='ETC (empirical)',
            color=colors['etc'], marker='o', capsize=3, linewidth=1.5)
ax.errorbar(gap, se_avg,  yerr=se_err,  label='Succ. Elim. (empirical)',
            color=colors['se'],  marker='s', capsize=3, linewidth=1.5)
ax.errorbar(gap, eg_avg,  yerr=eg_err,  label='ε-Greedy (empirical)',
            color=colors['eg'],  marker='^', capsize=3, linewidth=1.5)
ax.plot(gap, theo_etc, '--', color=colors['etc'], linewidth=1.5,
        label='ETC (theory)')
ax.plot(gap, theo_se,  '--', color=colors['se'],  linewidth=1.5,
        label='Succ. Elim. (theory)')
ax.plot(gap, theo_eg,  '--', color=colors['eg'],  linewidth=1.5,
        label='ε-Greedy (theory)')
ax.set_xlabel('Gap Δ', fontsize=12)
ax.set_ylabel('Expected Regret', fontsize=12)
ax.set_title('Empirical vs Theoretical Regret Bounds', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
