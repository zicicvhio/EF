import matplotlib.pyplot as plt
X=paddle.linspace(-5, 5,50)
c1=0
c2=0.5
c3=1
c4=100
y1=X*(1/(1+paddle.exp(-c1*X)))
y2=X*(1/(1+paddle.exp(-c2*X)))
y3=X*(1/(1+paddle.exp(-c3*X)))
y4=X*(1/(1+paddle.exp(-c4*X)))
plt.figure()
plt.plot(X,y1,'-',color='r',label='c1=0')
plt.plot(X,y2,'--',color='g',label='c2=0.5')
plt.plot(X,y3,':',color='b',label='c3=1')
plt.plot(X,y4,'4',color='k',label='c4=100')
#设置x的刻度间隔
new_ticks = np.linspace(-4,4,8)
plt.xticks=(new_ticks)
plt.yticks=(new_ticks)
plt.xlim((-4,4))
plt.ylim((-4,4))
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))
plt.legend(loc='lower right', fontsize='large')
plt.savefig('fw-logistic-swish.pdf')
plt.show()