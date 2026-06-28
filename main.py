import tkinter as tk
from PIL import Image, ImageDraw
from  network import *
from mnist_loader import *
from os.path  import join



input_path = './handwritten_digit_data/handwritten_digit_images'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


mnist_dataloader = MnistDataLoader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


x_train = [np.reshape(x, (784, 1)) for x in x_train]
y_train = [np.eye(10)[y].reshape(10, 1) for y in y_train]
x_test = [np.reshape(x, (784, 1)) for x in x_test]
y_test = [np.eye(10)[y].reshape(10, 1) for y in y_test]


net = network([784, 30, 10])
training_data = list(zip(x_train, y_train))
test_data = list(zip(x_test, y_test))
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


canvas_size = 280

root = tk.Tk()
root.title("Draw a Digit")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.pack()

image = Image.new("L", (canvas_size, canvas_size), 255)
draw = ImageDraw.Draw(image)

def paint(event):
    x, y = event.x, event.y
    r = 8

    canvas.create_oval(x-r, y-r, x+r, y+r,
                       fill="black", outline="black")

    draw.ellipse((x-r, y-r, x+r, y+r), fill=0)

canvas.bind("<B1-Motion>", paint)

def save():
    image.save("digit.png")
    print("Saved as digit.png")

btn = tk.Button(root, text="Save", command=save)
btn.pack()

root.mainloop()


from PIL import Image
import numpy as np

img = Image.open("digit.png").convert("L")
img = img.resize((28,28))
img = np.array(img)

# MNIST style preprocessing
img = 255 - img
img = img / 255.0

# shape according to your model
img = img.reshape(1, 28, 28, 1)
img = img.reshape(784, 1)

prediction = net.predict(img)
print(prediction)


