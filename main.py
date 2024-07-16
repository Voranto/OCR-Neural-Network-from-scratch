import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt

import pandas as pd 
import pygame
width = 500
height = 500
pygame.init()
screen = pygame.display.set_mode((width,height))
nnfs.init()

class CapaNeuronal:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1,n_neurons))
    
    #En el pase adelante, recibimos unos inputs y para cada input recibido lo multiplicamos por su weight, y lo sumamos todo junto al bias
    def forward_pass(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
        
    #En un backwards pass, los dvalues son los valores que obtenemos de la capa siguiente (mirando desde alante)
    def backwards_pass(self,dvalues):
        
        #La derivada de weight es suma de todos los dvalues por todos los inputs (x)
        self.dweights = np.dot(self.inputs.T,dvalues)
        
        #Como la derivada del bias es 1, basta con sumar dvalues
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        
        #La derivada del x (input), es la suma de todos los dvalues multiplicado por todos los weights
        self.dinputs = np.dot(dvalues,self.weights.T)
        
class RELU:
    def forward_pass(self,inputs):
        self.inputs = inputs
        #relu es max(0,z)
        self.output = np.maximum(0,self.inputs)

    def backward_pass(self,dvalues):
        #derivada de relu es 1 si z> 0 else 0
        
        self.dinputs = dvalues.copy()
        
        #hacemos copia de dvalues para alterarla segun relu
        
        self.dinputs[self.inputs<=0] = 0
    
#esta es la clase de activación del final        
class Softmax:
    
    #nos encargamos de convertir los outputs en las neuronas finales en una distribucion probabilistica
    def forward_pass(self,inputs):
        self.inputs = inputs
        #Exponenciamos y obtenemos probabilidad
        exponents = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        
        #Normalizamos todo dividiendo entre la suma
        self.output = exponents/np.sum(exponents,axis=1,keepdims=True)
        
    def backward_pass(self,dvalues):
        #hacemos array vacia
        self.dinputs = np.empty_like(dvalues)
        
        for idx ,(output,dvalue) in enumerate(zip(self.output,dvalues)):
            
            #Con reshape aplanamos el output a una sola columna
            output = output.reshape(-1,1)
            
            #matrix jacovian
            jacovian_matrix = np.diagflat(output)-np.dot(output,output.T)
            
            self.dinputs[idx] = np.dot(jacovian_matrix,dvalue)

#clase para Stochastic gradient descent
class SGD:
    
    #inicializamos learning rate como 1 pero decay y momentum lo dejamos vacio
    def __init__(self,learning_rate = 1, decay = 0.,momentum = 0.):
        self.learning_rate = learning_rate
        
        #para ir disminuyendo learning rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
    
    def update_learning_rate(self):
        if self.decay:
            #descenso gradual de la learning rate
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay*self.iterations))
    
    def update_rest(self,layer):
        #si hay momentum es distinto a si no lo hay
        if self.momentum:
            
            #si la layer no tiene matriz de momentums la creamos
            if not hasattr(layer,"weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            #algo de inercia que aporta al movimiento de weights y evita minimos locales
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate*layer.dweights
            
            layer.weight_momentums = weight_updates
            
            
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate*layer.dbiases
            
            layer.bias_momentums = bias_updates
        #si no hay momentum simplemente modificamos weights y bias dependiendo de la derivada multiplicado por la learning rate en negativo
        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            bias_updates = -self.current_learning_rate*layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    def next_iteration(self):
        self.iterations += 1


#clase sencilla loss    
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward_pass(output,y)
        
        data_loss = np.mean(sample_losses)
        
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward_pass(self,y_pred,y_true):
        
        samples = len(y_pred)
        
        #ajustar y_pred para evitar edge cases
        y_pred = np.clip(y_pred, 1e-7,1 - 1e-7)
        
        #correct confidences nos da el la probabilidad que habia del valor que terminó siendo el correcto, lo usamos para loss function
        #si la forma es de y true es un solo numero con el index
        if len(y_true.shape) == 1:
            correct_confidences = y_pred[range(samples),y_true]  
        #si la forma es una array con 0 como no y 1 como sí
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred*y_true,axis=1)
        #legaritmo de la probabilidad del correcto en negativo para pasarlo a positivo ya que logaritmos de x < 1 es negativo
        answer = -np.log(correct_confidences)
        return answer
    
    
    def backwards_pass(self,dvalues,y_true):
        
        samples = len(dvalues)
        
        labels = len(dvalues[0])
        
        #si los y true son un solo numero, lo convertimos en su array equivalente. 
        #Ejemplo, si y_true = 1 -> np.eye(labels) = [[1,0],[0,1]] por lo que np.eye(labels)[1] == [0,1]
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        #cálculo
        self.dinputs = -y_true/dvalues
        
        #normalización
        self.dinputs = self.dinputs / samples


#Clase que mezcla ambos la loss con la softmax para x7 optimización

class Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        #inicializamos ambas
        self.activation = Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
        
    def forward_pass(self,inputs,y_true):
        
        #activamos los inputs y lo igualamos al output
        self.activation.forward_pass(inputs)
        
        self.output = self.activation.output
        
        #calculamos pérdida
        return self.loss.calculate(self.output,y_true)
    
    def backward_pass(self,dvalues,y_true):
        samples = len(dvalues)
        
        #si tiene shape de valores individuales, escogemos el indice del mayor valor
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis = 1)
        
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        
        #normalizamos
        self.dinputs = self.dinputs/samples
        

#----------------------------------------------
#a partir de ahora procesamiento de datos y tal

TRAIN_DATA_archivo =  "training_data/train-images.idx3-ubyte"
TRAIN_LABELS_archivo =  "training_data/train-labels.idx1-ubyte"


def bytes_to_int(byte_data):
    if type(byte_data) == int:
        return byte_data
    else:
        return int.from_bytes(byte_data,"big")
    
    
def leer_imagenes(archivo,max_num_imgs = None):
    global count
    
    #matriz 3D que guarda todas las imagenes
    images = []
    with open(archivo, "rb") as f: #abrir el fichero  como f, y leerlo en binario ("rb")
        _  =f.read(4) #numero inutil (representa algo que no necesitamos)
        
        #los siguientes 12 bytes representan el numero de imagenes, el numero de filas y de columnas
        num_imgs = bytes_to_int(f.read(4))
        
        
        #max_num_imgs representa el numero de imagenes que queremos leer (predeterminado 60000)
        if max_num_imgs:
            num_imgs = max_num_imgs
            
        #el numero de filas y columnas son los proximos 8 bytes
        n_filas = bytes_to_int(f.read(4))
        n_columnas = bytes_to_int(f.read(4))
        for _ in range(num_imgs):
            image = np.array([]) #variable que guarda la imagen actual
            for _ in range(n_filas):
                row = np.array([])#variable que guarda la columna actual
                for _ in range(n_columnas):
                    pixel = f.read(1) #leemos el pixel actual de 8 bits y lo apendizamos a la row
                    if bytes_to_int(pixel) > 0:
                        row = np.append(row,255)
                    else:
                        row = np.append(row,0)
                image = np.append(image,row)#metemos la row en la image
            images.append(image)#metemos la image en el conjunto de images
    #devolvemos la array 3D
    return np.array(images)

def leer_etiquetas(archivo,n_max_labels = None):
    
    #variable que guarda todas las etiquetas
    labels = [] 
    
    
    with open(archivo, "rb") as f: #abrir el fichero archivo como f, y leerlo en binario ("rb")
        
        #numero inutil (representa algo que no necesitamos)
        _  =f.read(4)
        
        
        #los siguientes 12 bytes representan el numero de imagenes, el numero de filas y de columnas
        n_labels = bytes_to_int(f.read(4))
        #n_max_labels representa el numero de labels que queremos leer (predeterminado 60000)
        if n_max_labels:
            n_labels = n_max_labels
            
            
        #leemos el contenido de 1 byte en 1byte
        for _ in range(n_labels):
            label = f.read(1)
            labels.append(bytes_to_int(label))
            
    return np.array(labels)



X_train = leer_imagenes(TRAIN_DATA_archivo,40000)
X_train = X_train/255
Y_train = leer_etiquetas(TRAIN_LABELS_archivo,40000)


capa_input = CapaNeuronal(784,256)
capa_hidden1 = CapaNeuronal(256,128)
capa_hidden2 = CapaNeuronal(128,64)
capa_hidden3 = CapaNeuronal(64,32)
capa_output = CapaNeuronal(32,10)


activacion1 = RELU()
activacion2 = RELU()
activacion3 = RELU()
activacion4 = RELU()
activacion5 = Softmax_Loss_CategoricalCrossEntropy()
stop = False
optimizador = SGD(learning_rate= 0.01,decay=0.0001,momentum= 0.9)
for epoca in range(10000):
    if stop == True:
        break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.QUIT()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
           stop = True
    capa_input.forward_pass(X_train)
    activacion1.forward_pass(capa_input.output)
    capa_hidden1.forward_pass(activacion1.output)
    
    activacion2.forward_pass(capa_hidden1.output)
    
    capa_hidden2.forward_pass(activacion2.output)
    
    activacion3.forward_pass(capa_hidden2.output)
    
    capa_hidden3.forward_pass(activacion3.output)
    
    activacion4.forward_pass(capa_hidden3.output)
    
    capa_output.forward_pass(activacion4.output)
    
    
    loss = activacion5.forward_pass(capa_output.output,Y_train)
    
    prediccion = np.argmax(activacion5.output,axis=1)
    
    if len(Y_train.shape) == 2:
        y = np.argmax(Y_train,axis=1)
    accuracy = np.mean(prediccion == Y_train)
    print("Epoca: ",epoca," , accuracy: ",round(accuracy,6)," ,loss: ",round(loss,6), " ,learning rate: ",round(optimizador.current_learning_rate,6))
    activacion5.backward_pass(activacion5.output,Y_train)
    capa_output.backwards_pass(activacion5.dinputs)
    activacion4.backward_pass(capa_output.dinputs)
    capa_hidden3.backwards_pass(activacion4.dinputs)
    
    activacion3.backward_pass(capa_hidden3.dinputs)
    capa_hidden2.backwards_pass(activacion3.dinputs)
    activacion2.backward_pass(capa_hidden2.dinputs)
    capa_hidden1.backwards_pass(activacion2.dinputs)
    activacion1.backward_pass(capa_hidden1.dinputs)
    capa_input.backwards_pass(activacion1.dinputs)
    
    optimizador.update_learning_rate()
    optimizador.update_rest(capa_input)
    optimizador.update_rest(capa_hidden1)
    optimizador.update_rest(capa_hidden2)
    optimizador.update_rest(capa_hidden3)
    optimizador.update_rest(capa_output)
    optimizador.next_iteration()
print("finished")
#guardar todos los valores de los weights a un csv
DF = pd.DataFrame(capa_hidden1.weights) 
DF.to_csv("capa_hidden1.csv")
DF = pd.DataFrame(capa_hidden2.weights) 
DF.to_csv("capa_hidden2.csv")
DF = pd.DataFrame(capa_hidden3.weights) 
DF.to_csv("capa_hidden3.csv")
DF = pd.DataFrame(capa_output.weights) 
DF.to_csv("capa_output.csv")
DF = pd.DataFrame(capa_input.weights) 
DF.to_csv("capa_input.csv")
#guardar todos los biases
DF = pd.DataFrame(capa_hidden1.biases) 
DF.to_csv("capa_hidden1_b.csv")
DF = pd.DataFrame(capa_hidden2.biases) 
DF.to_csv("capa_hidden2_b.csv")
DF = pd.DataFrame(capa_hidden3.biases) 
DF.to_csv("capa_hidden3_b.csv")
DF = pd.DataFrame(capa_output.biases) 
DF.to_csv("capa_output_b.csv")
DF = pd.DataFrame(capa_input.biases) 
DF.to_csv("capa_input_b.csv")


        
        
