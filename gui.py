#libraries for GUI
from tkinter import *
from PIL import ImageGrab, ImageTk, Image

#libraries for preprocessing input data and loading pretrained models
import cv2
import numpy as np
from keras.models import load_model
import process_input

class App(object):
	#function for basic initializations
	def __init__(self):
		self.root = Tk()
		self.root.geometry("800x320")
		self.line_width = 20;
		self.color = 'white'
		self.canvas_size = 280
		
		self.canvas = Canvas(self.root, bg='black', width=self.canvas_size+5, height=self.canvas_size+5)
		self.canvas.place(x=0, y=0)
		
		self.setup()
		self.root.mainloop()

	#function converts content on canvas into a jpg image
	def take_snap(self):
			x=self.root.winfo_rootx()
			y=self.root.winfo_rooty()
			x1=x+self.canvas_size+5
			y1=y+self.canvas_size+5
			ImageGrab.grab().crop((x+5,y+5,x1,y1)).save("input_image.jpg")

	#this function takes care for events:
	#   1.  <B1-Motion> - trigger and active while drawing
	#   2.  <ButtonRelease-1> - trigger when prediction is needed
	def setup(self):
		self.old_x = None
		self.old_y = None
		self.canvas.bind('<B1-Motion>', self.write)
		self.canvas.bind('<ButtonRelease-1>', self.predict)

	#this function lets user write the digit
	#on canvas component of GUI
	def write(self, event):
		if self.old_x and self.old_y:
			self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
							   width=self.line_width, fill=self.color,
							   capstyle=ROUND, smooth=TRUE, splinesteps=36)
		self.old_x = event.x
		self.old_y = event.y

	
	#function responsible for:
	#   1. preprocesssing input data
	#   2. returns prediction from support vector classifier
	def predict_mlp(self,img):
		x_predict_mlp = img.reshape(1,784)
		prediction_mlp = model_mlp.predict(x_predict_mlp)
		return ('MLP prediction = '+  str(np.argmax(prediction_mlp)))

	#function responsible for:
	#   1. preprocesssing input data
	#   2. returns prediction from cnn model
	def predict_cnn(self,img):
		x_predict_cnn = img.reshape(-1,28,28,1)
		prediction_cnn = model_cnn.predict(x_predict_cnn)
		return ('CNN prediction='+ str(np.argmax(prediction_cnn, axis=None)))
	

	#function responsible:
	#   1. saving input image
	#   2. showing image used for model
	#   3. gets prediction from both models(MLP and CNN)
	#   4. shows the prediction on GUI
	def predict(self, event):
		#saving input_image from user and image_for_model
		self.take_snap()
		img = cv2.resize(cv2.imread('input_image.jpg',cv2.IMREAD_GRAYSCALE),(28,28))
		img2 = cv2.imread('input_image.jpg',cv2.IMREAD_GRAYSCALE)
		img2 = process_input.preprocess(img2)    
		cv2.imwrite("image_for_model.jpg",cv2.resize(img2,(self.canvas_size,self.canvas_size)))
		
		#showing image_for_model to GUI
		img_2 = ImageTk.PhotoImage(Image.open('image_for_model.jpg'))
		panel_cnn = Label(self.root, image = img_2, width=280, height=280)
		panel_cnn.image = img_2
		panel_cnn.place(x=300 , y = 0)
		
		#for GUI - predicted result from svc model
		result_mlp = self.predict_mlp(img2)
		label_mlp = Label(self.root,text = result_mlp)   #creates label for image on window 
		label_mlp.place(x = 630, y = 100)
		
		#for GUI - predicted result from cnn model
		result_cnn = self.predict_cnn(img2)
		label_cnn = Label(self.root,text = result_cnn)   #creates label for image on window 
		label_cnn.place(x = 630, y = 140)

		self.canvas.delete("all")
		self.old_x, self.old_y = None, None
	

if __name__ == '__main__':
	model_cnn = load_model('training/cnn/model/digit_classifier_cnn_1.h5')
	model_mlp = load_model('training/mlp/trained_model/digit_classifier_mlp_1.h5') 
	ge = App()