import matplotlib.pyplot as plt
labels = 'Computer Science', 'Foreign Languges','Analytical Chemistry', 'Education', 'Humanities', 'Physics', 'Biology', 'Math and Statistics', 'Engineering'
sizes = [21, 4, 7, 7, 8, 9, 10, 15, 19]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red', 'purple', '#f280de', 'orange', 'green']
explode = (0,0,0,0,0,0,0,0,0.1)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',colors=colors)
plt.axis('equal')
plt.show()