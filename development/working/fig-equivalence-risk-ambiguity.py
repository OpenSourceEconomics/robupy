import matplotlib.pyplot as plt
import numpy as np

from robupy.auxiliary import get_entropic_risk_measure
from robupy.auxiliary import get_multiplier_evaluation
from robupy.auxiliary import get_worst_case_outcome
from robupy.config import SMALL_FLOAT

# We fix the maximum number of risk aversion and the number of points
np.random.seed(12)
max_theta, num_points = 2, 10
# TODO: Integrate 0.00


v = np.random.normal(size=num_points, scale=10)
q = np.random.uniform(low=0.0, size=num_points)
q = q / np.sum(q)

# TODO: V is defined as cost, negative utility

theta_grid = np.linspace(0.0, max_theta, endpoint=True, num=num_points)

rslt = dict()
for label in ['entropic', 'hard', 'soft']:
    rslt[label] = list()

for i in range(num_points):

    theta = theta_grid[i]

    if theta == 0.0:
        gamma = np.inf
    else:
        gamma = 1.0 / theta

    rslt['hard'].append(get_worst_case_outcome(v, q, theta, is_cost=False))
    rslt['soft'].append(get_multiplier_evaluation(v, q, gamma))
    rslt['entropic'].append(-get_entropic_risk_measure(v, q, theta))

# Simple data to display in various forms
x = theta_grid
y = rslt['entropic']

## Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

ax1.plot(x, rslt['hard'])
#ax3.set_xlabel(r'$\theta')
ax1.set_title('Maxmin')
#a = ax3.get_xticks().tolist()
#a[5] = r'$\theta^{max}$'
#ax3.set_xticklabels(a)


ax2.plot(x, rslt['soft'])
ax2.set_title('Multiplier')
#a = ax1.get_xticks().tolist()
#a[5] = r'$\theta^{max}$'
#ax1.set_xticklabels(a)

#

ax3.plot(x, rslt['entropic'])
#ax1.set_xlabel(r'$\theta')
ax3.set_title('Exponential')
#a = ax1.get_xticks().tolist()
#a[5] = r'$1 / \theta^{max}$'
#ax1.set_xticklabels(a)


plt.savefig('fig-equivalence-risk-ambiguity')
