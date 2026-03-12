import matplotlib.pyplot as plt
import numpy as np

# time steps
t = np.arange(30)

# cumulative reward
reward_do_nothing = [
-0.81,-2.48,-3.10,-5.53,-12.63,-18.30,-22.35,-25.59,-28.83,-31.07,
-36.74,-45.65,-49.51,-57.61,-60.04,-64.09,-68.95,-77.86,-81.91,-83.53,
-87.58,-96.49,-105.40,-109.26,-113.31,-118.98,-122.22,-127.08,-134.37,-144.09
]

reward_rule = [
-3.137,-3.375,-4.063,-4.526,-4.764,-5.227,-5.465,-7.153,-11.065,-12.528,
-13.991,-15.454,-15.917,-16.155,-16.393,-16.631,-16.869,-18.332,-20.020,
-20.258,-21.721,-23.184,-23.422,-23.660,-23.898,-24.361,-24.599,-25.062,
-25.525,-26.975
]

reward_throttle = [
-0.335,-0.67,-1.005,-1.34,-1.835,-2.17,-2.505,-3.0,-3.335,-3.67,
-4.005,-4.34,-4.675,-5.01,-5.345,-5.68,-6.015,-6.35,-6.685,-8.34,
-8.675,-9.01,-9.345,-9.68,-10.175,-10.51,-10.845,-11.18,-11.515,-11.85
]

# engagement
eng_do_nothing = [
1,8,10,13,23,30,35,39,43,47,54,65,71,81,84,89,95,106,111,113,
118,129,140,146,151,158,162,168,177,189
]

eng_rule = [
4,4,6,7,7,8,8,10,13,14,15,16,17,17,17,17,17,18,20,20,
21,22,22,22,22,23,23,24,25,26
]

eng_throttle = [
0,0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,4,
4,4,4,4,5,5,5,5,5,5
]

# reports
rep_do_nothing = [
1,4,5,8,17,24,29,33,37,40,47,58,63,73,76,81,87,98,103,105,
110,121,132,137,142,149,153,159,168,180
]

rep_rule = [
2,2,2,2,2,2,2,3,6,7,8,9,9,9,9,9,9,10,11,11,
12,13,13,13,13,13,13,13,13,14
]

rep_throttle = [
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
1,1,1,1,1,1,1,1,1,1
]


plt.style.use("dark_background")

t = np.arange(30)

# colors
c1 = "#00E5FF"   # cyan
c2 = "#FFD54F"   # yellow
c3 = "#FF6E6E"   # red

fig, axes = plt.subplots(
    3,1,
    figsize=(10,7),
    sharex=True
)

fig.patch.set_facecolor("black")

# reward
axes[0].plot(t,reward_do_nothing,color=c1,label="Do Nothing",linewidth=2)
axes[0].plot(t,reward_rule,color=c2,label="Rule Policy",linewidth=2)
axes[0].plot(t,reward_throttle,color=c3,label="Always Throttle",linewidth=2)
axes[0].set_title("Cumulative Reward",color="white")
axes[0].grid(True,alpha=0.3)

# engagement
axes[1].plot(t,eng_do_nothing,color=c1,label="Do Nothing",linewidth=2)
axes[1].plot(t,eng_rule,color=c2,label="Rule Policy",linewidth=2)
axes[1].plot(t,eng_throttle,color=c3,label="Always Throttle",linewidth=2)
axes[1].set_title("Total Engagement",color="white")
axes[1].grid(True,alpha=0.3)

# reports
axes[2].plot(t,rep_do_nothing,color=c1,label="Do Nothing",linewidth=2)
axes[2].plot(t,rep_rule,color=c2,label="Rule Policy",linewidth=2)
axes[2].plot(t,rep_throttle,color=c3,label="Always Throttle",linewidth=2)
axes[2].set_title("Total Reports",color="white")
axes[2].set_xlabel("Time Step",color="white")
axes[2].grid(True,alpha=0.3)

for ax in axes:
    ax.legend(facecolor="black", edgecolor="white")
    ax.tick_params(colors="white")

plt.tight_layout()
plt.show()