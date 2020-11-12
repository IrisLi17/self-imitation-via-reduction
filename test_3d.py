import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
a = np.zeros(100)
b = np.random.uniform(0,1,100)
c = np.random.uniform(0,0.7,100)
d = np.random.uniform(-3,3,100)
matplotlib.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(c,b,d,marker='o',s=0.4,c='tab:orange',label='state')
ax.scatter(a,b,d,marker='*',s=0.4,c='tab:red',label='truth')
ax.set_title('2d mean value plot for subgoal_dist_to_goal')
ax.set_xlabel('subgoal_mean_value_to_goal_dist')
ax.set_ylabel('dist_to_goal')
ax.set_zlabel('value')
fig.legend(loc='lower right')
fig.savefig('test_3d_y.png')
plt.close(fig)
