from matplotlib import pyplot as plt
x_axis = ['(384-384-384)', '(384-384-384) tf-lite', '(192-256-128)', '(192-256-128) tf-lite', '(128-128-128)', '(128-128-128) tf-lite']
y_axis = [43, 39.6, 27.1, 23.7, 21.9, 18.5]
plt.bar(x_axis, y_axis)
plt.title('Figure 4: Model sizes on disk in MB')
plt.xlabel('Model architecture')
plt.ylabel('Size in MB (Mega Bytes)')
plt.ylim(top = 45, bottom = 0)
plt.show()