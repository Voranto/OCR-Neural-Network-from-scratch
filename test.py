import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt
import math
import pandas as pd 

import time
import pygame
draw = False
if input("Do you desire to draw the number, or to pass the Neural Network through a standardized test with 10k characters (draw/test): ") == "draw":
    draw = True 
class CapaNeuronal:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1,n_neurons))
    
    #En el pase adelante, recibimos unos inputs y para cada input recibido lo multiplicamos por su weight, y lo sumamos todo junto al bias
    def forward_pass(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
        
        
class RELU:
    def forward_pass(self,inputs):
        self.inputs = inputs
        #relu es max(0,z)
        self.output = np.maximum(0,self.inputs)

    
#esta es la clase de activación del final        
class Softmax:
    
    #nos encargamos de convertir los outputs en las neuronas finales en una distribucion probabilistica
    def forward_pass(self,inputs):
        self.inputs = inputs
        #Exponenciamos y obtenemos probabilidad
        exponents = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        
        #Normalizamos todo dividiendo entre la suma
        self.output = exponents/np.sum(exponents,axis=1,keepdims=True)
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




#Clase que mezcla ambos la loss con la softmax para x7 optimización

class Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        #inicializamos ambas
        self.activation = Softmax()
        
    def forward_pass(self,inputs):
        
        #activamos los inputs y lo igualamos al output
        self.activation.forward_pass(inputs)
        
        self.output = self.activation.output



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

optimizador = SGD(learning_rate=0.01,decay=0.001,momentum=0.01)

df =pd.read_csv('test_final2/capa_input.csv').to_numpy()
capa_input.weights = np.delete(df, (0), axis=1)
df =pd.read_csv('test_final2/capa_hidden1.csv').to_numpy()
capa_hidden1.weights = np.delete(df, (0), axis=1)
df =pd.read_csv('test_final2/capa_hidden2.csv').to_numpy()
capa_hidden2.weights = np.delete(df, (0), axis=1)
df =pd.read_csv('test_final2/capa_hidden3.csv').to_numpy()
capa_hidden3.weights = np.delete(df, (0), axis=1)
df =pd.read_csv('test_final2/capa_output.csv').to_numpy()
capa_output.weights = np.delete(df, (0), axis=1)

df =pd.read_csv('test_final2/capa_input_b.csv').to_numpy()
capa_input.biases = np.delete(df, (0), axis=1)
df =pd.read_csv('test_final2/capa_hidden1_b.csv').to_numpy()
capa_hidden1.biases = np.delete(df, (0), axis=1)
df =pd.read_csv('test_final2/capa_hidden2_b.csv').to_numpy()
capa_hidden2.biases = np.delete(df, (0), axis=1)
df =pd.read_csv('test_final2/capa_hidden3_b.csv').to_numpy()
capa_hidden3.biases = np.delete(df, (0), axis=1)
df =pd.read_csv('test_final2/capa_output_b.csv').to_numpy()
capa_output.biases = np.delete(df, (0), axis=1)




#--------------------------------------------PYGAME----------------------------
pygame.init()

#Inicializamos tambien el sound engine de pygame, para los sonidos
pygame.mixer.init()


#Unas variables clave para altura y anchura de la pantalla
width = 1000
height = 784

#esta variable puede parecer redundante, ya que es igual que la anchura, pero me permite poder variar las dimensiones de la pantalla 
#sin arruinar todo el canvas y funciones de dibujo
grid_size = 784

#renderizar la pantalla
screen = pygame.display.set_mode((width,height))



#Título
pygame.display.set_caption("Proyecto Final Tico. Luca Siegel Moreno 2ºBH")



#reloj/framerate
clock = pygame.time.Clock()
image_array = [] 
for i in range(0,grid_size+28,28):
    pygame.draw.line(screen,"white",(0,i),(784,i))



#Mismo bucle en el eje X
#Aprovechamos uno de estos loops para llenar la matriz "image_array", con todo 0s, que es el valor predeterminado(corresponde al color negro)
for j in range(0,grid_size,28):
    image_array.append([0]*28)
    pygame.draw.line(screen,"white",(j,0),(j,784))
pygame.draw.line(screen,"white",(grid_size,0),(grid_size,784))


def drawerase(x,y,color_name,color_value,brush_size = 2):
    pygame.draw.rect(screen,color_name,(x-x%28,y-y%28,28,28)) 
    image_array[math.trunc(y/28)][math.trunc(x/28)] = color_value
    #si la brush size es mayor que 1, implica que tenemos que dibujar más pixeles ademas de ese único ( la cruz de 3x3)
    if brush_size > 1:
        
        #pixel de la derecha
        if math.trunc(x/28)+1 < 28: 
            pygame.draw.rect(screen,color_name,(x-x%28+28,y-y%28,28,28)) 
            image_array[math.trunc(y/28)][math.trunc(x/28)+1] = color_value
        
        #pixel izquierda
        if math.trunc(x/28)-1 >= 0:  
            pygame.draw.rect(screen,color_name,(x-x%28-28,y-y%28,28,28))
            image_array[math.trunc(y/28)][math.trunc(x/28)-1] = color_value
        
        #pixel debajo
        if math.trunc(y/28)+1 < 28: 
            pygame.draw.rect(screen,color_name,(x-x%28,y-y%28+28,28,28)) 
            image_array[math.trunc(y/28)+1][math.trunc(x/28)] = color_value
        
        
        #pixel encima
        if math.trunc(y/28)-1 >= 0: 
            pygame.draw.rect(screen,color_name,(x-x%28,y-y%28-28,28,28)) 
            image_array[math.trunc(y/28)-1][math.trunc(x/28)] = color_value
            
        #si la brush size es mayor que 2, es decir, 3, hay que dibujar el cuadrado de 3x3 entero
        if brush_size > 2:
            
            #pixel abajo derecha
            if math.trunc(x/28)+1 < 28 and math.trunc(y/28)+1 < 28: 
                pygame.draw.rect(screen,color_name,(x-x%28+28,y-y%28+28,28,28)) 
                image_array[math.trunc(y/28)+1][math.trunc(x/28)+1] = color_value
            
            #pixel arriba izquierda
            if math.trunc(x/28)-1 >= 0 and math.trunc(y/28)-1 >= 0:  
                pygame.draw.rect(screen,color_name,(x-x%28-28,y-y%28-28,28,28))
                image_array[math.trunc(y/28)-1][math.trunc(x/28)-1] = color_value
            
            #pixel abajo izquierda
            if math.trunc(y/28)+1 < 28 and math.trunc(x/28)-1 >= 0: 
                pygame.draw.rect(screen,color_name,(x-x%28-28,y-y%28+28,28,28)) 
                image_array[math.trunc(y/28)+1][math.trunc(x/28)-1] = color_value
            
            
            #pixel arriba derecha
            if math.trunc(y/28)-1 >= 0 and math.trunc(x/28)+1 < 28: 
                pygame.draw.rect(screen,color_name,(x-x%28+28,y-y%28-28,28,28)) 
                image_array[math.trunc(y/28)-1][math.trunc(x/28)+1] = color_value

button_pressed = False
pos = pygame.mouse.get_pos()
def guess(X_test):
    capa_input.forward_pass(X_test)
    activacion1.forward_pass(capa_input.output)
    capa_hidden1.forward_pass(activacion1.output)
    
    activacion2.forward_pass(capa_hidden1.output)
    
    capa_hidden2.forward_pass(activacion2.output)
    
    activacion3.forward_pass(capa_hidden2.output)
    
    capa_hidden3.forward_pass(activacion3.output)
    
    activacion4.forward_pass(capa_hidden3.output)
    
    capa_output.forward_pass(activacion4.output)
    
    activacion5.forward_pass(capa_output.output)
    
    top_ans = {}
    preds = activacion5.output[0]
    for i in range(len(activacion5.output[0])):
        top_ans[float(activacion5.output[0,i])] = i
    return top_ans,preds

def clearcanvasfunction():
    global image_array
    #limpiamos el canvas entero dibujando un cuadrado de las coordenadas de la grid entera que lo ocupe tood
    pygame.draw.rect(screen, "black",(0,0,784,784))
    
    #vaciamos la array con los valores de los colores de los pixeles
    image_array = []
    
    #Rehacemos las grid lines, ya que se han borrado con el cuadrado dibujado previamente
    for i in range(0,grid_size+28,28):
        pygame.draw.line(screen,"white",(0,i),(784,i))
    for j in range(0,grid_size,28):
        #recompletamos la image_array
        image_array.append([0]*28)
        pygame.draw.line(screen,"white",(j,0),(j,784))


font = pygame.font.Font(None, 50) 
mode = "draw"
if draw:
    print("To draw, use left click. To erase, use right click, and to clear the canvas, press on the right side of the canvas, where the predictions are displayed")
while True:
    if draw == False:
        break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.QUIT()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            #Si apretamos el boton de click "hacia abajo", que la variable de estado se active
            if event.button == 1:
                mode = "draw"
            elif event.button == 3:
                mode = "erase"
            button_pressed = True
            if pos[0] > 784:
                clearcanvasfunction()
            
        elif event.type == pygame.MOUSEBUTTONUP:
            #Si apretamos el boton de click "hacia abajo", que la variable de estado se active
            button_pressed = False
            for i in range(0,grid_size+28,28):
                pygame.draw.line(screen,"white",(0,i),(784,i))
            for j in range(0,grid_size,28):
                pygame.draw.line(screen,"white",(j,0),(j,784))
            pygame.draw.line(screen,"white",(grid_size,0),(grid_size,784))
    pos = pygame.mouse.get_pos()
    if pos[1] < 784 and pos[0] < 784 and button_pressed == True:
        if mode =="draw":
            drawerase(pos[0],pos[1],"white",255)
        elif mode =="erase":
            drawerase(pos[0],pos[1],"black",0)
        
    
    pred_dict,preds = guess((np.array(image_array)/255).flatten())
    
    preds.sort()
    pygame.draw.rect(screen,"black",(784,0,784,784))
    for i in range(1,10):
        pred_text = font.render(str(pred_dict[preds[-i]])+": "+str(round(preds[-i]*100,3))+"%", True, (255, 255, 0))
        screen.blit(pred_text, (790,10+50*i))
    pygame.display.update()


#---------------testing space

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

def bytes_to_int(byte_data):
    if type(byte_data) == int:
        return byte_data
    else:
        return int.from_bytes(byte_data,"big")


TEST_DATA_archivo =  "training_data/t10k-images.idx3-ubyte"
TEST_LABELS_archivo =  "training_data/t10k-labels.idx1-ubyte"

X_test = leer_imagenes(TEST_DATA_archivo)
Y_test = leer_etiquetas(TEST_LABELS_archivo)


capa_input.forward_pass(X_test)
activacion1.forward_pass(capa_input.output)
capa_hidden1.forward_pass(activacion1.output)

activacion2.forward_pass(capa_hidden1.output)

capa_hidden2.forward_pass(activacion2.output)

activacion3.forward_pass(capa_hidden2.output)

capa_hidden3.forward_pass(activacion3.output)

activacion4.forward_pass(capa_hidden3.output)

capa_output.forward_pass(activacion4.output)

activacion5.forward_pass(capa_output.output)
    
prediccion = np.argmax(activacion5.output,axis=1)
print(np.mean(prediccion == Y_test))