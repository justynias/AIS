import neurolab
 
with open('kohonen1.mat') as f:
    data = f.read()
 
kohonen = data.split('\n')
d = []
 
for entry in kohonen:
    tmp = entry.split(' ')
    try:
        d.append([float(tmp[0]), float(tmp[1])])
    except ValueError:
        break
 
net = neurolab.net.newc([[0.0, 1.0],[0.0, 1.0]], 4)
error = net.train(d, epochs=200, show=20)