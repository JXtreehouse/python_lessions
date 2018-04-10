import numpy as np
import matplotlib.pyplot as plt

N = 7
winnersplot = (142.6, 125.3, 62.0, 81.0, 145.6, 319.4, 178.1 )
ind = np.arange(N)
width = 0.35
fig, ax = plt.subplots()
winners = ax.bar(ind, winnersplot, width, color='#ffad00')
print(winners)

nomineesplot = (109.4, 94.8, 60.7, 44.6, 116.9,262.5,102.0)
nominees = ax.bar(ind + width, nomineesplot, width, color='#9b3c38')

# add some text for labels ,title and axes ticks

ax.set_xticks(ind+width)
ax.set_xticklabels(('小明', '小红', '小凡', '小钱', '小刘', '小赵', '小文'))
ax.legend((winners[0], nominees[0]),('奥斯卡金奖得住','奥斯卡得住提名'))

def autolabel(rects):
    # attach some text labels
   for rect in rects:
       height = rect.get_height()
       hcap = "$" + str(height) + "M"
       ax.text(rect.get_x() + rect.get_width()/2. ,height, hcap,ha = 'center',va='bottom',rotation='horizontal')

autolabel(winners)
autolabel(nominees)

plt.show()
