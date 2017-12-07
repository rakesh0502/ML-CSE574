import matplotlib.pyplot as plt


####################### Logistic Regression Graph #######################
############################## MNIST ####################################

mx = [0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.1,0.3,0.5]
my = [0.7908, 0.8333, 0.8489, 0.8646, 0.8719, 0.8939, 0.9008, 0.9084, 0.9175, 0.9204]
my1 = [0.8339, 0.8681, 0.8798, 0.8886, 0.8925, 0.9088, 0.9141, 0.9171, 0.9237, 0.9215]
my2 = [0.8497, 0.8797, 0.8898, 0.898, 0.9016, 0.9146, 0.9183, 0.9215, 0.924, 0.9182]
my3 = [0.8658, 0.8891, 0.8981, 0.9057, 0.9082, 0.9175, 0.92, 0.9221, 0.9216, 0.9239]

# epochs number blue = 1, red =3, yellow = 5, green =8

plt.figure(1)
plt.plot(mx, my, 'b^', label ="epoch =1", linestyle='-')
plt.plot(mx, my1, 'r^', label ="epoch =3", linestyle='-')
plt.plot(mx, my2, 'y^',label = "epoch =5", linestyle='-')
plt.plot(mx, my3, 'g^',label = "epoch = 8", linestyle='-')
plt.legend()
plt.axis([-0.5, 0.55, 0.75, 1])
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('MNIST Logistic Regression')
plt.show()

############################## USPS ####################################

ux = [0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.1,0.3,0.5]
uy = [0.2869, 0.3197, 0.32725, 0.33315, 0.3375, 0.3528, 0.3622, 0.3688, 0.37345, 0.36975]
uy1 = [0.3183, 0.33585, 0.3436, 0.35065, 0.3538, 0.36725, 0.36915, 0.37035, 0.35985, 0.35675]
uy2 = [0.32755, 0.34285, 0.3511, 0.35805, 0.36125, 0.3696, 0.36855, 0.3673, 0.35745, 0.35275]
uy3 = [0.3338, 0.35015, 0.3591, 0.36395, 0.36665, 0.3681, 0.3677, 0.36185, 0.3584, 0.35205]

# epochs number blue = 1, red =3, yellow = 5, green =8

plt.figure(2)
plt.plot(ux, uy, 'b^', label ="epoch =1", linestyle='-')
plt.plot(ux, uy1, 'r^', label ="epoch =3", linestyle='-')
plt.plot(ux, uy2, 'y^',label = "epoch =5", linestyle='-')
plt.plot(ux, uy3, 'g^',label = "epoch = 8", linestyle='-')
plt.legend()
plt.axis([-0.5, 0.55, 0.25, 0.4])
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('USPS Logistic Regression')
plt.show()


############################ MNIST Vs USPS ####################################

# epoch 5 yellow is MNIST and blue is USPS

plt.figure(3)
plt.plot(mx, my2, 'y^', label = "MNIST ", linestyle='-')
plt.plot(ux, uy2,'b^',label = "USPS", linestyle='-')
plt.legend()
plt.axis([-0.5, 0.55, 0.25, 1])
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('USPS VS MNIST Logistic Regression')
plt.show()


############################ SNN MNIST VS USPS ################################

lr = [0.003, 0.01, 0.1, 1]
epochs = [10, 20, 30]
nodes = [256, 512, 1024, 1568]

# best learning rate 0.01
# MNIST data for best learning rate

m_snnAccuracy10 = [0.9415, 0.9499, 0.955, 0.9553]
m_snnAccuracy20 = [0.9532, 0.9609, 0.9638, 0.9685]
m_snnAccuracy30 = [0.9645, 0.9663, 0.964, 0.9646]

u_snnAccuracy10 = [0.39485, 0.4136, 0.4407, 0.46855]
u_snnAccuracy20 = [0.40825, 0.4578, 0.48035, 0.48705]
u_snnAccuracy30 = [0.463, 0.48355, 0.5121, 0.5134]


plt.figure(4)
plt.plot(nodes, m_snnAccuracy10, 'b^',label='epoch 10', linestyle='-')
plt.plot(nodes, m_snnAccuracy20, 'r^',label="epoch 20", linestyle='-')
plt.plot(nodes, m_snnAccuracy30, 'y^',label="epoch 30",linestyle='-')
plt.legend()
plt.axis([200, 1600, 0.9, 1])
plt.xlabel('Number of nodes in Hidden Layer')
plt.ylabel('Accuracy')
plt.title('MNIST SNN')

plt.show()



plt.figure(5)
plt.plot(nodes, u_snnAccuracy10, 'b^',label='epoch 10', linestyle='-')
plt.plot(nodes, u_snnAccuracy20, 'r^',label="epoch 20", linestyle='-')
plt.plot(nodes, u_snnAccuracy30, 'y^',label="epoch 30", linestyle='-')
plt.legend()

plt.axis([200, 1600, 0.40, 0.55])
plt.xlabel('Number of nodes in Hidden Layer')
plt.ylabel('Accuracy')
plt.title(' USPS SNN')
plt.show()


plt.figure(6)
plt.plot(nodes, m_snnAccuracy20, 'b^',label="MNIST",linestyle ='-')
plt.plot(nodes, u_snnAccuracy20, 'r^',label="USPS",linestyle='-')
plt.legend()
plt.axis([200, 1600, 0.40, 1])
plt.xlabel('Number of nodes in Hidden Layer')
plt.ylabel('Accuracy')
plt.title('MNIST Vs USPS SNN')
plt.show()
